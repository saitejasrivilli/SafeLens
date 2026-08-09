from pathlib import Path

import torch

from safelens.models.vision.clip.config import HeadConfig
from safelens.models.vision.clip.head import ClassificationHead
from safelens.models.vision.clip.infer import (
    benchmark_encoder_latency,
    benchmark_end_to_end_latency,
    benchmark_head_latency,
    benchmark_preprocessing_latency,
)


def test_benchmark_preprocessing_latency(tiny_clip_processor, sample_image):
    bench = benchmark_preprocessing_latency(
        tiny_clip_processor, sample_image, warmup_iterations=1, measured_iterations=3
    )
    assert bench.stage == "preprocessing"
    assert bench.p50_ms > 0
    assert bench.p95_ms >= bench.p50_ms
    assert bench.measured_iterations == 3
    assert bench.warmup_iterations == 1
    assert bench.batch_size == 1


def test_benchmark_encoder_latency(tiny_clip_model, tiny_clip_processor, sample_image):
    inputs = tiny_clip_processor(images=sample_image, return_tensors="pt")
    bench = benchmark_encoder_latency(
        tiny_clip_model, inputs["pixel_values"], "cpu", warmup_iterations=1, measured_iterations=3
    )
    assert bench.stage == "clip_encoder"
    assert bench.device == "cpu"
    assert bench.p50_ms > 0


def test_benchmark_head_latency():
    head = ClassificationHead(HeadConfig(hidden_dim=8, dropout=0.0), embed_dim=64)
    sample_embedding = torch.randn(1, 64)
    bench = benchmark_head_latency(
        head, sample_embedding, warmup_iterations=1, measured_iterations=3
    )
    assert bench.stage == "classification_head"
    assert bench.p50_ms >= 0


def test_benchmark_end_to_end_latency(
    tmp_path: Path, tiny_clip_model, tiny_clip_processor, sample_image
):
    image_path = tmp_path / "sample.jpg"
    sample_image.save(image_path, format="JPEG")
    head = ClassificationHead(
        HeadConfig(hidden_dim=8, dropout=0.0), embed_dim=tiny_clip_model.config.projection_dim
    )
    bench = benchmark_end_to_end_latency(
        tiny_clip_processor,
        tiny_clip_model,
        head,
        image_path,
        "cpu",
        warmup_iterations=1,
        measured_iterations=3,
    )
    assert bench.stage == "end_to_end"
    assert bench.throughput_examples_per_second > 0
