import torch

from safelens.models.vision.clip.config import HeadConfig, ImageTrainingConfig
from safelens.models.vision.clip.head import ClassificationHead
from safelens.models.vision.clip.train import train_head

EMBED_DIM = 16


def _synthetic_data(n: int, embed_dim: int = EMBED_DIM, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(n, embed_dim, generator=g)
    labels = torch.randint(0, 2, (n,), generator=g)
    return embeddings, labels


def test_train_head_returns_loadable_state_dict():
    train_emb, train_labels = _synthetic_data(40)
    dev_emb, dev_labels = _synthetic_data(10, seed=1)

    config = ImageTrainingConfig(epochs=3, batch_size=8, early_stopping_patience=3)
    head_config = HeadConfig(hidden_dim=8, dropout=0.1)

    result = train_head(
        train_emb, train_labels, dev_emb, dev_labels, config, head_config, embed_dim=EMBED_DIM
    )

    head = ClassificationHead(head_config, embed_dim=EMBED_DIM)
    head.load_state_dict(result.best_state_dict)  # must not raise
    head.eval()
    with torch.no_grad():
        probs = torch.softmax(head(dev_emb), dim=-1)[:, 1]
    assert probs.shape == (10,)
    assert all(0.0 <= p <= 1.0 for p in probs.tolist())
    assert torch.isfinite(probs).all()


def test_train_head_is_deterministic_given_seed():
    train_emb, train_labels = _synthetic_data(40)
    dev_emb, dev_labels = _synthetic_data(10, seed=1)
    config = ImageTrainingConfig(epochs=3, batch_size=8, seed=42, early_stopping_patience=3)
    head_config = HeadConfig(hidden_dim=8, dropout=0.1)

    result_a = train_head(
        train_emb, train_labels, dev_emb, dev_labels, config, head_config, embed_dim=EMBED_DIM
    )
    result_b = train_head(
        train_emb, train_labels, dev_emb, dev_labels, config, head_config, embed_dim=EMBED_DIM
    )

    for key in result_a.best_state_dict:
        torch.testing.assert_close(result_a.best_state_dict[key], result_b.best_state_dict[key])


def test_class_weights_reported_when_enabled():
    train_emb, train_labels = _synthetic_data(40)
    dev_emb, dev_labels = _synthetic_data(10, seed=1)
    config = ImageTrainingConfig(epochs=2, batch_size=8, use_class_weighting=True)
    head_config = HeadConfig(hidden_dim=8, dropout=0.1)

    result = train_head(
        train_emb, train_labels, dev_emb, dev_labels, config, head_config, embed_dim=EMBED_DIM
    )
    assert result.class_weights is not None
    assert len(result.class_weights) == 2


def test_class_weights_none_when_disabled():
    train_emb, train_labels = _synthetic_data(40)
    dev_emb, dev_labels = _synthetic_data(10, seed=1)
    config = ImageTrainingConfig(epochs=2, batch_size=8, use_class_weighting=False)
    head_config = HeadConfig(hidden_dim=8, dropout=0.1)

    result = train_head(
        train_emb, train_labels, dev_emb, dev_labels, config, head_config, embed_dim=EMBED_DIM
    )
    assert result.class_weights is None
