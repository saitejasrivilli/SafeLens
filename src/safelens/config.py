"""Configuration loading: YAML file + environment variable overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SAFELENS_")

    log_level: str = "INFO"
    environment: Literal["dev", "staging", "production"] = "dev"


def load_settings(config_file: Path | None = None) -> Settings:
    config_file = config_file or (CONFIG_DIR / "base.yaml")
    file_values: dict[str, Any] = {}
    if config_file.exists():
        file_values = yaml.safe_load(config_file.read_text()) or {}
    return Settings(**file_values)
