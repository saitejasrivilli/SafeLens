"""Stage-by-stage latency benchmarking for the fusion model: image
preprocessing, CLIP encoder, text tokenization, AraBERT encoder, fusion
head, and end-to-end. Reuses the LatencyBenchmark shape/measurement
discipline from Phase 5A/5B (batch size 1, warmup + measured iterations).
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, PreTrainedModel, PreTrainedTokenizerBase

from safelens.models.text_multilingual.encoder import encode_texts
from safelens.models.vision.clip.encoder import encode_images
from safelens.models.vision.clip.head import ClassificationHead
from safelens.models.vision.clip.infer import LatencyBenchmark, _summarize


@torch.no_grad()
def benchmark_fusion_end_to_end_latency(
    image_processor: CLIPImageProcessor,
    image_model: CLIPModel,
    text_tokenizer: PreTrainedTokenizerBase,
    text_model: PreTrainedModel,
    head: ClassificationHead,
    raw_image_root: Path,
    sample_image_path_rel: str,
    sample_text: str,
    max_seq_length: int,
    device: str,
    warmup_iterations: int = 5,
    measured_iterations: int = 50,
) -> LatencyBenchmark:
    head.eval()
    image_path = raw_image_root / sample_image_path_rel

    def _run() -> None:
        image = Image.open(image_path).convert("RGB")
        pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"]
        img_emb = encode_images(image_model, pixel_values, device)

        encodings = text_tokenizer(
            sample_text,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt",
        )
        txt_emb = encode_texts(
            text_model, encodings["input_ids"], encodings["attention_mask"], device
        )

        fused = torch.cat([txt_emb, img_emb], dim=1)
        head(fused)

    for _ in range(warmup_iterations):
        _run()

    latencies = []
    for _ in range(measured_iterations):
        t0 = time.perf_counter()
        _run()
        latencies.append((time.perf_counter() - t0) * 1000)

    result = _summarize("multimodal_end_to_end", latencies, device)
    return LatencyBenchmark(**{**result.__dict__, "warmup_iterations": warmup_iterations})


@torch.no_grad()
def benchmark_fusion_head_latency(
    head: ClassificationHead,
    sample_fused_embedding: torch.Tensor,
    warmup_iterations: int = 5,
    measured_iterations: int = 50,
) -> LatencyBenchmark:
    head.eval()
    for _ in range(warmup_iterations):
        head(sample_fused_embedding)

    latencies = []
    for _ in range(measured_iterations):
        t0 = time.perf_counter()
        head(sample_fused_embedding)
        latencies.append((time.perf_counter() - t0) * 1000)

    result = _summarize("fusion_head", latencies, "cpu")
    return LatencyBenchmark(**{**result.__dict__, "warmup_iterations": warmup_iterations})
