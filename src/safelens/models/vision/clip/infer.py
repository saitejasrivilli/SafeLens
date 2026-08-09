"""Stage-by-stage latency benchmarking: preprocessing, frozen-CLIP encoder,
classification head, and end-to-end. Batch size 1 (single-request serving
scenario), matching the same measurement discipline used in Phase 4."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel

from safelens.models.vision.clip.encoder import encode_images
from safelens.models.vision.clip.head import ClassificationHead


@dataclass(frozen=True)
class LatencyBenchmark:
    stage: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_examples_per_second: float
    batch_size: int
    warmup_iterations: int
    measured_iterations: int
    device: str


def _summarize(stage: str, latencies_ms: list[float], device: str) -> LatencyBenchmark:
    arr = np.array(latencies_ms)
    return LatencyBenchmark(
        stage=stage,
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        throughput_examples_per_second=1000.0 / arr.mean(),
        batch_size=1,
        warmup_iterations=0,  # filled in by caller
        measured_iterations=len(latencies_ms),
        device=device,
    )


def benchmark_preprocessing_latency(
    processor: CLIPImageProcessor,
    sample_image: Image.Image,
    warmup_iterations: int = 5,
    measured_iterations: int = 50,
) -> LatencyBenchmark:
    for _ in range(warmup_iterations):
        processor(images=sample_image, return_tensors="pt")

    latencies = []
    for _ in range(measured_iterations):
        t0 = time.perf_counter()
        processor(images=sample_image, return_tensors="pt")
        latencies.append((time.perf_counter() - t0) * 1000)

    result = _summarize("preprocessing", latencies, "cpu")
    return LatencyBenchmark(**{**result.__dict__, "warmup_iterations": warmup_iterations})


def benchmark_encoder_latency(
    model: CLIPModel,
    sample_pixel_values: torch.Tensor,
    device: str,
    warmup_iterations: int = 5,
    measured_iterations: int = 50,
) -> LatencyBenchmark:
    for _ in range(warmup_iterations):
        encode_images(model, sample_pixel_values, device)

    latencies = []
    for _ in range(measured_iterations):
        t0 = time.perf_counter()
        encode_images(model, sample_pixel_values, device)
        latencies.append((time.perf_counter() - t0) * 1000)

    result = _summarize("clip_encoder", latencies, device)
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
    processor: CLIPImageProcessor,
    model: CLIPModel,
    head: ClassificationHead,
    image_path: Path,
    device: str,
    warmup_iterations: int = 5,
    measured_iterations: int = 50,
) -> LatencyBenchmark:
    head.eval()

    def _run() -> None:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
        embedding = encode_images(model, pixel_values, device)
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
