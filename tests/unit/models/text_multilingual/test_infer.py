import torch

from safelens.models.text_multilingual.infer import (
    benchmark_encoder_latency,
    benchmark_end_to_end_latency,
    benchmark_head_latency,
    benchmark_tokenization_latency,
)
from safelens.models.vision.clip.config import HeadConfig
from safelens.models.vision.clip.head import ClassificationHead


def test_benchmark_tokenization_latency(tiny_text_tokenizer):
    bench = benchmark_tokenization_latency(
        tiny_text_tokenizer, "sample text", 16, warmup_iterations=1, measured_iterations=3
    )
    assert bench.stage == "tokenization"
    assert bench.p50_ms >= 0
    assert bench.measured_iterations == 3


def test_benchmark_encoder_latency(tiny_text_model, tiny_text_tokenizer):
    enc = tiny_text_tokenizer(
        "sample text", truncation=True, padding="max_length", max_length=16, return_tensors="pt"
    )
    bench = benchmark_encoder_latency(
        tiny_text_model,
        enc["input_ids"],
        enc["attention_mask"],
        "cpu",
        warmup_iterations=1,
        measured_iterations=3,
    )
    assert bench.stage == "text_encoder"
    assert bench.device == "cpu"


def test_benchmark_head_latency(tiny_text_model):
    head = ClassificationHead(
        HeadConfig(hidden_dim=8, dropout=0.0), embed_dim=tiny_text_model.config.hidden_size
    )
    sample_embedding = torch.randn(1, tiny_text_model.config.hidden_size)
    bench = benchmark_head_latency(
        head, sample_embedding, warmup_iterations=1, measured_iterations=3
    )
    assert bench.stage == "classification_head"


def test_benchmark_end_to_end_latency(tiny_text_model, tiny_text_tokenizer):
    head = ClassificationHead(
        HeadConfig(hidden_dim=8, dropout=0.0), embed_dim=tiny_text_model.config.hidden_size
    )
    bench = benchmark_end_to_end_latency(
        tiny_text_tokenizer,
        tiny_text_model,
        head,
        "sample text",
        16,
        "cpu",
        warmup_iterations=1,
        measured_iterations=3,
    )
    assert bench.stage == "end_to_end"
    assert bench.throughput_examples_per_second > 0
