import math

import torch

from safelens.models.text_multilingual.encoder import encode_texts


def test_encode_texts_shape_and_determinism(tiny_text_model, tiny_text_tokenizer):
    enc = tiny_text_tokenizer(
        ["hello world"], truncation=True, padding="max_length", max_length=16, return_tensors="pt"
    )
    hidden = tiny_text_model.config.hidden_size

    emb_a = encode_texts(tiny_text_model, enc["input_ids"], enc["attention_mask"], "cpu")
    emb_b = encode_texts(tiny_text_model, enc["input_ids"], enc["attention_mask"], "cpu")

    assert emb_a.shape == (1, hidden)
    assert torch.equal(emb_a, emb_b)
    assert not torch.isnan(emb_a).any()
    assert all(math.isfinite(v) for v in emb_a.flatten().tolist())


def test_encoder_params_are_frozen(tiny_text_model):
    assert all(not p.requires_grad for p in tiny_text_model.parameters())


def test_mean_pooling_respects_attention_mask(tiny_text_model, tiny_text_tokenizer):
    short = tiny_text_tokenizer(
        ["hi"], truncation=True, padding="max_length", max_length=16, return_tensors="pt"
    )
    long = tiny_text_tokenizer(
        ["hi there this is a longer sentence than the other one"],
        truncation=True,
        padding="max_length",
        max_length=16,
        return_tensors="pt",
    )
    emb_short = encode_texts(tiny_text_model, short["input_ids"], short["attention_mask"], "cpu")
    emb_long = encode_texts(tiny_text_model, long["input_ids"], long["attention_mask"], "cpu")
    # Different attention masks (different real token counts) should generally
    # produce different pooled embeddings.
    assert not torch.equal(emb_short, emb_long)
