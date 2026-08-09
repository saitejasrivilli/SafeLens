from safelens.data.preprocessing.leakage import check_leakage
from safelens.data.preprocessing.split import SplitConfig, split_dataset


def _dataset(make_example_fn, n=40):
    examples = []
    for i in range(n):
        toxicity = 0.9 if i % 2 == 0 else 0.1
        examples.append(make_example_fn(f"comment number {i}", toxicity=toxicity))
    return examples


def test_split_sizes_match_config(make_example_fn):
    examples = _dataset(make_example_fn)
    splits = split_dataset(examples, SplitConfig(seed=42, train_frac=0.7, val_frac=0.15))
    total = sum(len(v) for v in splits.values())
    assert total == len(examples)
    assert len(splits["train"]) == 28  # exact: 0.7 * 40, stratify divides evenly at this size
    # val/test split rounding on the stratified remainder can be off by one example
    assert abs(len(splits["validation"]) - 6) <= 1
    assert abs(len(splits["test"]) - 6) <= 1


def test_split_is_deterministic(make_example_fn):
    examples = _dataset(make_example_fn)
    splits_a = split_dataset(examples, SplitConfig(seed=42))
    splits_b = split_dataset(examples, SplitConfig(seed=42))
    for name in ("train", "validation", "test"):
        ids_a = [ex.content_id for ex in splits_a[name]]
        ids_b = [ex.content_id for ex in splits_b[name]]
        assert ids_a == ids_b


def test_split_has_no_leakage(make_example_fn):
    examples = _dataset(make_example_fn)
    splits = split_dataset(examples, SplitConfig(seed=42))
    assert check_leakage(splits).is_clean


def test_split_is_stratified(make_example_fn):
    examples = _dataset(make_example_fn)
    splits = split_dataset(examples, SplitConfig(seed=42))
    for name, exs in splits.items():
        toxic_frac = sum(1 for e in exs if e.labels.toxicity >= 0.5) / len(exs)
        assert abs(toxic_frac - 0.5) <= 0.15, f"{name} split not stratified: {toxic_frac}"
