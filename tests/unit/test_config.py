from safelens.config import load_settings


def test_load_settings_defaults():
    settings = load_settings()
    assert settings.environment in {"dev", "staging", "production"}
    assert settings.log_level == "INFO"
