"""Batched inference and latency benchmarking for the fine-tuned model."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.no_grad()
def predict_proba(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_seq_length: int,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()
    model.to(device)  # type: ignore[arg-type]  # PreTrainedModel.to() str overload
    probs: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encodings = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**encodings).logits
        batch_probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        probs.append(batch_probs)
    return np.concatenate(probs)


@dataclass(frozen=True)
class LatencyBenchmark:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_examples_per_second: float
    batch_size: int
    warmup_iterations: int
    measured_iterations: int
    sequence_length: int
    device: str


def benchmark_model_only_latency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sample_text: str,
    max_seq_length: int,
    device: str,
    warmup_iterations: int = 5,
    measured_iterations: int = 50,
) -> LatencyBenchmark:
    """Model-only latency: excludes tokenization, batch size 1 (single-request
    serving scenario)."""
    model.eval()
    model.to(device)  # type: ignore[arg-type]  # PreTrainedModel.to() str overload
    encoding = tokenizer(
        sample_text,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        for _ in range(warmup_iterations):
            model(**encoding)

        latencies_ms = []
        for _ in range(measured_iterations):
            t0 = time.perf_counter()
            model(**encoding)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies_ms)
    return LatencyBenchmark(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        throughput_examples_per_second=1000.0 / arr.mean(),
        batch_size=1,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        sequence_length=max_seq_length,
        device=device,
    )


def benchmark_end_to_end_latency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    sample_text: str,
    max_seq_length: int,
    device: str,
    warmup_iterations: int = 5,
    measured_iterations: int = 50,
) -> LatencyBenchmark:
    """End-to-end latency: tokenization + model forward pass, batch size 1."""
    model.eval()
    model.to(device)  # type: ignore[arg-type]  # PreTrainedModel.to() str overload

    def _run() -> None:
        encoding = tokenizer(
            sample_text,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            model(**encoding)

    for _ in range(warmup_iterations):
        _run()

    latencies_ms = []
    for _ in range(measured_iterations):
        t0 = time.perf_counter()
        _run()
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies_ms)
    return LatencyBenchmark(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        throughput_examples_per_second=1000.0 / arr.mean(),
        batch_size=1,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
        sequence_length=max_seq_length,
        device=device,
    )
