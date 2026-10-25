"""Make the GPU steps work on smaller / older NVIDIA cards too (e.g. a 6 GB Quadro P3000, Pascal):
  * amp()            -- bfloat16 autocast where the card has it (RTX 30-series and newer), plain float32 otherwise
                        (Pascal / Turing have no bfloat16: PyTorch stops with an error)
  * raft_flow(...)   -- optical flow over frame pairs; on "out of memory" the batch is split in half and retried

Test hooks: JOI_NO_BF16=1 pretends the card has no bfloat16; JOI_GPU_GB=<n> pretends the card has n GB.
"""
from __future__ import annotations

import contextlib
import os


def _torch():
    import torch
    return torch


def bf16_ok() -> bool:
    torch = _torch()
    if os.environ.get("JOI_NO_BF16") == "1" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported()


def amp():
    """Autocast context for model inference."""
    torch = _torch()
    return torch.autocast("cuda", dtype=torch.bfloat16) if bf16_ok() else contextlib.nullcontext()


def limit_memory():
    """JOI_GPU_GB=<n>: cap this process to n GB of GPU memory (to test small cards on a big one)."""
    gb = os.environ.get("JOI_GPU_GB")
    torch = _torch()
    if gb and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(float(gb) / total, 1.0), 0)


_SAID = False


def raft_flow(model, a, b, updates: int):
    """RAFT flow for frame pairs (a[i] -> b[i]); halves the batch on out-of-memory until it fits."""
    torch = _torch()
    try:
        return model(a, b, num_flow_updates=updates)[-1]
    except torch.cuda.OutOfMemoryError:
        if len(a) <= 1:
            raise
        torch.cuda.empty_cache()
        global _SAID
        if not _SAID:
            print(f"    (graphics card memory full - working in smaller batches)", flush=True)
            _SAID = True
        h = len(a) // 2
        return torch.cat([raft_flow(model, a[:h], b[:h], updates), raft_flow(model, a[h:], b[h:], updates)], 0)
