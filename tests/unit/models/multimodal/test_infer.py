from pathlib import Path

import torch

from safelens.models.multimodal.infer import (
    benchmark_fusion_end_to_end_latency,
    benchmark_fusion_head_latency,
)
from safelens.models.vision.clip.config import HeadConfig
from safelens.models.vision.clip.head import ClassificationHead


def test_benchmark_fusion_head_latency():
    head = ClassificationHead(HeadConfig(hidden_dim=8, dropout=0.0), embed_dim=32)
    sample = torch.randn(1, 32)
    bench = benchmark_fusion_head_latency(head, sample, warmup_iterations=1, measured_iterations=3)
    assert bench.stage == "fusion_head"
    assert bench.p50_ms >= 0


def test_benchmark_fusion_end_to_end_latency(
    tmp_path: Path,
    tiny_clip_processor,
    tiny_clip_model,
    tiny_text_tokenizer,
    tiny_text_model,
    sample_image,
):
    image_path = tmp_path / "sample.jpg"
    sample_image.save(image_path, format="JPEG")

    embed_dim = tiny_clip_model.config.projection_dim + tiny_text_model.config.hidden_size
    head = ClassificationHead(HeadConfig(hidden_dim=8, dropout=0.0), embed_dim=embed_dim)

    bench = benchmark_fusion_end_to_end_latency(
        tiny_clip_processor,
        tiny_clip_model,
        tiny_text_tokenizer,
        tiny_text_model,
        head,
        tmp_path,
        "sample.jpg",
        "sample text",
        16,
        "cpu",
        warmup_iterations=1,
        measured_iterations=3,
    )
    assert bench.stage == "multimodal_end_to_end"
    assert bench.throughput_examples_per_second > 0
