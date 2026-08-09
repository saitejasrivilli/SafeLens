from safelens.utils.device import detect_device


def test_detect_device_returns_valid_backend():
    info = detect_device()
    assert info.device in {"cuda", "mps", "cpu"}
    assert isinstance(info.torch_available, bool)
    assert info.reason
