import torch

from safelens.models.text.deberta.dataset import ToxicityDataset

TEXTS = ["you are wonderful", "this is a threat, watch out", "have a nice day"]
LABELS = [0, 1, 0]
CONTENT_IDS = ["a", "b", "c"]


def test_dataset_length_and_item_shape(tiny_tokenizer):
    ds = ToxicityDataset(TEXTS, LABELS, CONTENT_IDS, tiny_tokenizer, max_seq_length=16)
    assert len(ds) == 3
    item = ds[0]
    assert item["input_ids"].shape == (16,)
    assert item["attention_mask"].shape == (16,)
    assert item["labels"] == torch.tensor(0)


def test_dataset_tokenization_is_deterministic(tiny_tokenizer):
    ds_a = ToxicityDataset(TEXTS, LABELS, CONTENT_IDS, tiny_tokenizer, max_seq_length=16)
    ds_b = ToxicityDataset(TEXTS, LABELS, CONTENT_IDS, tiny_tokenizer, max_seq_length=16)
    for i in range(len(ds_a)):
        assert torch.equal(ds_a[i]["input_ids"], ds_b[i]["input_ids"])
        assert torch.equal(ds_a[i]["attention_mask"], ds_b[i]["attention_mask"])


def test_dataset_padding_respects_max_seq_length(tiny_tokenizer):
    ds = ToxicityDataset(TEXTS, LABELS, CONTENT_IDS, tiny_tokenizer, max_seq_length=8)
    for i in range(len(ds)):
        assert ds[i]["input_ids"].shape == (8,)
