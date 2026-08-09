"""Stage-by-stage latency benchmarking: tokenization, frozen text encoder,
classification head, and end-to-end. Reuses the LatencyBenchmark shape from
the Phase 5A vision module -- same measurement discipline (batch size 1,
warmup + measured iterations), architecture-agnostic dataclass."""

from __future__ import annotations

import time

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from safelens.models.text_multilingual.encoder import encode_texts
from safelens.models.vision.clip.head import ClassificationHead
from safelens.models.vision.clip.infer import LatencyBenchmark, _summarize


def benchmark_tokenization_latency(
    tokenizer: PreTrainedTokenizerBase,
    sample_text: str,
    max_seq_length: int,
    warmup_iterations: int = 5,
    measured_iterations: int = 50,
) -> LatencyBenchmark:
    def _run() -> None:
        tokenizer(
            sample_text,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt",
        )

    for _ in range(warmup_iterations):
        _run()

    latencies = []
    for _ in range(measured_iterations):
        t0 = time.perf_counter()
        _run()
        latencies.append((time.perf_counter() - t0) * 1000)

    result = _summarize("tokenization", latencies, "cpu")
    return LatencyBenchmark(**{**result.__dict__, "warmup_iterations": warmup_iterations})


def benchmark_encoder_latency(
    model: PreTrainedModel,
    sample_input_ids: torch.Tensor,
    sample_attention_mask: torch.Tensor,
    device: str,
    warmup_iterations: int = 5,
    measured_iterations: int = 50,
) -> LatencyBenchmark:
    for _ in range(warmup_iterations):
        encode_texts(model, sample_input_ids, sample_attention_mask, device)

    latencies = []
    for _ in range(measured_iterations):
        t0 = time.perf_counter()
        encode_texts(model, sample_input_ids, sample_attention_mask, device)
        latencies.append((time.perf_counter() - t0) * 1000)

    result = _summarize("text_encoder", latencies, device)
    return LatencyBenchmark(**{**result.__dict__, "warmup_iterations": warmup_iterations})


@torch.no_grad()
def benchmark_head_latency(
    head: ClassificationHead,
    sample_embedding: torch.Tensor,
    warmup_iterations: int = 5,
    measured_iterations: int = 50,
) -> LatencyBenchmark:
    head.eval()
    for _ in range(warmup_iterations):
        head(sample_embedding)

    latencies = []
    for _ in range(measured_iterations):
        t0 = time.perf_counter()
        head(sample_embedding)
        latencies.append((time.perf_counter() - t0) * 1000)

    result = _summarize("classification_head", latencies, "cpu")
    return LatencyBenchmark(**{**result.__dict__, "warmup_iterations": warmup_iterations})


@torch.no_grad()
def benchmark_end_to_end_latency(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    head: ClassificationHead,
    sample_text: str,
    max_seq_length: int,
    device: str,
    warmup_iterations: int = 5,
    measured_iterations: int = 50,
) -> LatencyBenchmark:
    head.eval()

    def _run() -> None:
        encodings = tokenizer(
            sample_text,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt",
        )
        embedding = encode_texts(model, encodings["input_ids"], encodings["attention_mask"], device)
        head(embedding)

    for _ in range(warmup_iterations):
        _run()

    latencies = []
    for _ in range(measured_iterations):
        t0 = time.perf_counter()
        _run()
        latencies.append((time.perf_counter() - t0) * 1000)

    result = _summarize("end_to_end", latencies, device)
    return LatencyBenchmark(**{**result.__dict__, "warmup_iterations": warmup_iterations})
