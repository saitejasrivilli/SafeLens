"""Device detection: CUDA -> MPS -> CPU. Never hard-code a backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    device: str  # "cuda", "mps", or "cpu"
    torch_available: bool
    reason: str


def detect_device() -> DeviceInfo:
    try:
        import torch
    except ImportError:
        return DeviceInfo(device="cpu", torch_available=False, reason="torch not installed")

    if torch.cuda.is_available():
        return DeviceInfo(device="cuda", torch_available=True, reason="torch.cuda.is_available()")
    if torch.backends.mps.is_available():
        return DeviceInfo(
            device="mps", torch_available=True, reason="torch.backends.mps.is_available()"
        )
    return DeviceInfo(device="cpu", torch_available=True, reason="no CUDA or MPS backend found")
