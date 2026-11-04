"""vLLM v0.32.0 synthetic CPU KV connector.

Experiment-only connector, not real semantic KV reuse. A forced block-aligned
prefix is populated with synthetic K/V from CPU DRAM before real residual
prefill. Scheduler reports an external host hit. No hash lookup or offload.

IMPORTANT: Only single-rank dense full-attention models with standard LBHNC
[B,2,block,H,D] paged KV layers are supported. Fail closed otherwise.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole,
)
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.utils.gpu_sync_debug import gpu_sync_allowed


@dataclass
class LoadRequest:
    request_id: str
    block_ids: list[int]
    tokens: int


@dataclass
class LoadMetadata(KVConnectorMetadata):
    loads: list[LoadRequest] = field(default_factory=list)


class SyntheticCPUConnector(KVConnectorBase_V1):
    """Declare forced prefix length, transfer synthetic CPU KV into live cache."""

    _cache_hit_source = CacheHitSource.HOST

    @property
    def requires_kv_delivery(self) -> bool:
        return False

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        self.block_size = int(vllm_config.cache_config.block_size)
        self.forced_hit_tokens = int(extra.get("forced_hit_tokens", 0))
        self.pinned = bool(extra.get("pinned", True))
        self.expected_layers = int(
            getattr(vllm_config.model_config.hf_config,
                    "num_hidden_layers", 0) or 0
        )
        self.output_dir = pathlib.Path(
            extra.get("metrics_dir", "/state/partition/fyzhou/results/cpu-kv")
        )
        self._pending: dict[str, int] = {}
        self._block_ids: dict[str, list[int]] = {}
        self._counter = 0
        # Persist CPU-resident source KV across measured requests.
        # The first request of each cell is excluded as warm-up.
        self._cpu_layers: dict[str, torch.Tensor] = {}
        if self.forced_hit_tokens < 0 or self.forced_hit_tokens % self.block_size:
            raise ValueError("forced_hit_tokens must be >=0 and block aligned")
        if (vllm_config.parallel_config.world_size != 1
                or len(kv_cache_config.kv_cache_groups) != 1):
            raise ValueError("Only one GPU and one full-attention KV cache group supported")
        if getattr(vllm_config.model_config, "is_multimodal_model", False):
            raise ValueError("Multimodal models not supported")
        if role == KVConnectorRole.WORKER:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        # At least one token must be computed to produce output.
        prompt_len = len(request.prompt_token_ids or [])
        upper = max(0, ((prompt_len - 1) // self.block_size) * self.block_size)
        hit = min(self.forced_hit_tokens, upper)
        if num_computed_tokens > 0:
            return 0, False
        return hit, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if num_external_tokens:
            self._pending[request.request_id] = int(num_external_tokens)

    def build_connector_meta(self, scheduler_output):
        metadata = LoadMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            rid = new_req.req_id
            tokens = self._pending.pop(rid, 0)
            if tokens:
                need = tokens // self.block_size
                block_ids = list(new_req.block_ids[0][:need])
                if len(block_ids) != need:
                    raise RuntimeError(f"Insufficient GPU KV blocks for {rid}")
                metadata.loads.append(LoadRequest(rid, block_ids, tokens))
        if self._pending:
            raise RuntimeError("Unscheduled forced CPU KV loads: " +
                               repr(self._pending))
        return metadata

    def start_load_kv(self, forward_context, **kwargs: Any):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, LoadMetadata) or not metadata.loads:
            return
        layers = []
        for layer_name, layer in forward_context.no_compile_layers.items():
            kv = getattr(layer, "kv_cache", None)
            if kv is not None and isinstance(kv, torch.Tensor) and kv.numel():
                layers.append((layer_name, kv))
        if not layers:
            raise RuntimeError("No registered KV tensors to populate")
        if self.expected_layers and len(layers) != self.expected_layers:
            raise RuntimeError(
                f"Expected {self.expected_layers} KV layers, got {len(layers)}"
            )
        # Store request-level metrics and per-layer timings in worker process.
        for load in metadata.loads:
            t0 = time.perf_counter_ns()
            n_blocks = len(load.block_ids)
            block_ids_cpu = torch.tensor(load.block_ids, dtype=torch.long)
            records = []
            total_bytes = 0
            for layer_name, dst in layers:
                # vLLM v0.32.0 standard non-MLA cache:
                # [num_blocks, 2, block_size, num_kv_heads, head_size].
                if (dst.ndim != 5 or dst.shape[1] != 2
                        or dst.shape[2] != self.block_size
                        or dst.device.type != "cuda"):
                    raise RuntimeError(
                        f"Unsupported KV layout for {layer_name}: {tuple(dst.shape)}"
                    )
                if max(load.block_ids) >= dst.shape[0]:
                    raise RuntimeError(f"KV block ID out of bounds: {layer_name}")
                shape = (n_blocks, *dst.shape[1:])
                prep0 = time.perf_counter_ns()
                host = self._cpu_layers.get(layer_name)
                if host is None or tuple(host.shape) != shape:
                    host = torch.empty(shape, dtype=dst.dtype, device="cpu",
                                       pin_memory=self.pinned)
                    # Initialize once: measured requests reuse the same CPU KV.
                    host.fill_(0.125)
                    self._cpu_layers[layer_name] = host
                index = block_ids_cpu.to(dst.device, non_blocking=False)
                stage = torch.empty(shape, dtype=dst.dtype, device=dst.device)
                with gpu_sync_allowed():
                    torch.cuda.synchronize(dst.device)
                prep_ms = (time.perf_counter_ns() - prep0) / 1e6

                # CUDA events isolate H2D and block-scatter device activity.
                h0, h1, s1 = (torch.cuda.Event(enable_timing=True)
                              for _ in range(3))
                wall0 = time.perf_counter_ns()
                h0.record()
                stage.copy_(host, non_blocking=self.pinned)
                h1.record()
                dst.index_copy_(0, index, stage)
                s1.record()
                with gpu_sync_allowed():
                    s1.synchronize()
                wall_ms = (time.perf_counter_ns() - wall0) / 1e6
                h2d_ms = h0.elapsed_time(h1)
                scatter_ms = h1.elapsed_time(s1)
                nbytes = host.numel() * host.element_size()
                total_bytes += nbytes
                records.append(dict(layer=str(layer_name), bytes=nbytes,
                                    prepare_ms=prep_ms, h2d_cuda_ms=h2d_ms,
                                    scatter_cuda_ms=scatter_ms,
                                    load_wall_ms=wall_ms))
                del index, stage
            self._counter += 1
            result = dict(
                kind="synthetic_cpu_kv_live_vllm", request_id=load.request_id,
                requested_hit_tokens=self.forced_hit_tokens,
                injected_hit_tokens=load.tokens, block_size=self.block_size,
                gpu_block_count=n_blocks, layers=len(layers),
                bytes=total_bytes, pinned=self.pinned,
                total_worker_load_wall_ms=(time.perf_counter_ns()-t0)/1e6,
                sum_layer_h2d_cuda_ms=sum(x["h2d_cuda_ms"] for x in records),
                sum_layer_scatter_cuda_ms=sum(x["scatter_cuda_ms"] for x in records),
                sum_layer_prepare_ms=sum(x["prepare_ms"] for x in records),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                layer_metrics=records)
            filename = self.output_dir / (
                f"kvload-{os.getpid()}-{self._counter:05d}.json")
            filename.write_text(json.dumps(result, indent=2) + "\n")

    def wait_for_layer_load(self, layer_name):
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        return

    def wait_for_save(self):
        return
