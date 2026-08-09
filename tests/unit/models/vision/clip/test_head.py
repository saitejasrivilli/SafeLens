import torch

from safelens.models.vision.clip.config import HeadConfig
from safelens.models.vision.clip.head import ClassificationHead


def test_head_output_shape():
    head = ClassificationHead(HeadConfig(hidden_dim=16, dropout=0.1), embed_dim=64)
    x = torch.randn(5, 64)
    logits = head(x)
    assert logits.shape == (5, 2)


def test_head_deterministic_in_eval_mode():
    head = ClassificationHead(HeadConfig(hidden_dim=16, dropout=0.5), embed_dim=64)
    head.eval()
    x = torch.randn(3, 64)
    out_a = head(x)
    out_b = head(x)
    assert torch.equal(out_a, out_b)  # dropout disabled in eval -> deterministic


def test_head_produces_finite_logits():
    head = ClassificationHead(HeadConfig(hidden_dim=8, dropout=0.0), embed_dim=64)
    head.eval()
    x = torch.randn(4, 64)
    logits = head(x)
    assert torch.isfinite(logits).all()
