import math

from safelens.models.text.deberta.infer import benchmark_model_only_latency, predict_proba

TEXTS = ["you are wonderful", "this is a threat", "have a nice day", "i hate this"]


def test_predict_proba_shape_and_range(tiny_model, tiny_tokenizer):
    probs = predict_proba(tiny_model, tiny_tokenizer, TEXTS, max_seq_length=16, device="cpu")
    assert probs.shape == (len(TEXTS),)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert all(not math.isnan(p) and math.isfinite(p) for p in probs)


def test_predict_proba_batching_does_not_change_results(tiny_model, tiny_tokenizer):
    probs_one_batch = predict_proba(
        tiny_model, tiny_tokenizer, TEXTS, max_seq_length=16, device="cpu", batch_size=32
    )
    probs_small_batches = predict_proba(
        tiny_model, tiny_tokenizer, TEXTS, max_seq_length=16, device="cpu", batch_size=1
    )
    assert probs_one_batch.tolist() == probs_small_batches.tolist()


def test_benchmark_model_only_latency_reports_expected_fields(tiny_model, tiny_tokenizer):
    bench = benchmark_model_only_latency(
        tiny_model,
        tiny_tokenizer,
        TEXTS[0],
        max_seq_length=16,
        device="cpu",
        warmup_iterations=1,
        measured_iterations=3,
    )
    assert bench.p50_ms > 0
    assert bench.p95_ms >= bench.p50_ms
    assert bench.p99_ms >= bench.p95_ms
    assert bench.batch_size == 1
    assert bench.warmup_iterations == 1
    assert bench.measured_iterations == 3
    assert bench.device == "cpu"
