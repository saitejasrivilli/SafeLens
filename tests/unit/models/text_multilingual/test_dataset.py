import torch

from safelens.models.text_multilingual.dataset import extract_embeddings


class _NoImageExample:
    """Duck-typed stand-in for MultimodalExample whose `image_path`
    attribute raises if ever accessed -- proves extract_embeddings truly
    never touches image/filename information."""

    def __init__(self, example_id: str, text: str, hate_label: int):
        self.example_id = example_id
        self.text = text
        self.hate_label = hate_label

    @property
    def image_path(self):
        raise AssertionError("text-only pipeline must never access .image_path")


def test_extract_embeddings_never_touches_image(tiny_text_tokenizer, tiny_text_model):
    examples = [
        _NoImageExample("id1", "some Arabic-ish placeholder text", 0),
        _NoImageExample("id2", "another sentence here", 1),
    ]

    embeddings, labels, ids = extract_embeddings(
        examples, tiny_text_tokenizer, tiny_text_model, "cpu", max_seq_length=16, batch_size=2
    )

    hidden = tiny_text_model.config.hidden_size
    assert embeddings.shape == (2, hidden)
    assert labels.tolist() == [0, 1]
    assert ids == ["id1", "id2"]


def test_extract_embeddings_batching_matches_single(tiny_text_tokenizer, tiny_text_model):
    examples = [
        _NoImageExample("id1", "first sentence", 0),
        _NoImageExample("id2", "second sentence", 1),
    ]

    batched, _, _ = extract_embeddings(
        examples, tiny_text_tokenizer, tiny_text_model, "cpu", max_seq_length=16, batch_size=2
    )
    one_at_a_time, _, _ = extract_embeddings(
        examples, tiny_text_tokenizer, tiny_text_model, "cpu", max_seq_length=16, batch_size=1
    )
    torch.testing.assert_close(batched, one_at_a_time)
