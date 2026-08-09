import json
from pathlib import Path

from safelens.data.manifest import build_manifest, sha256_examples, write_manifest


def test_manifest_hash_deterministic(make_example_fn):
    a = make_example_fn("comment a")
    b = make_example_fn("comment b")
    assert sha256_examples([a, b]) == sha256_examples([b, a])  # order-independent


def test_manifest_hash_changes_with_content(make_example_fn):
    a = make_example_fn("comment a")
    b = make_example_fn("comment b")
    c = make_example_fn("comment c")
    assert sha256_examples([a, b]) != sha256_examples([a, c])


def test_build_manifest_fields(tmp_path: Path, make_example_fn):
    raw_path = tmp_path / "pool.jsonl"
    raw_path.write_text('{"text": "x"}\n')

    splits = {
        "train": [make_example_fn("a"), make_example_fn("b")],
        "validation": [make_example_fn("c")],
        "test": [make_example_fn("d")],
    }
    manifest = build_manifest(
        dataset_name="google/civil_comments",
        dataset_version="google/civil_comments@train:0-4",
        source_url="https://huggingface.co/datasets/google/civil_comments",
        license_id="cc0-1.0",
        retrieval_date="2026-08-09T00:00:00+00:00",
        raw_path=raw_path,
        splits=splits,
        seed=42,
    )
    for key in [
        "dataset_name",
        "dataset_version",
        "source_url",
        "license",
        "retrieval_date",
        "raw_data_hash",
        "processed_data_hash",
        "num_records",
        "label_distribution_fraction_ge_0.5",
        "split_sizes",
        "preprocessing_version",
        "random_seed",
    ]:
        assert key in manifest

    assert manifest["num_records"] == 4
    assert manifest["split_sizes"] == {"train": 2, "validation": 1, "test": 1}
    assert manifest["random_seed"] == 42

    out_path = tmp_path / "manifest.json"
    write_manifest(manifest, out_path)
    assert json.loads(out_path.read_text()) == manifest
