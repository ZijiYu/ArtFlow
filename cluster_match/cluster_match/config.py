from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_FINAL_CONFIG_PATH = Path("/Users/ken/MM/Pipeline/final_version/config.yaml")
DEFAULT_DATASET_PATH = Path("/Users/ken/MM/Pipeline/eval_v3/artifacts/dataset_pilot_30.jsonl")
DEFAULT_BASELINE_DIR = Path("/Users/ken/MM/Pipeline/cluster_match/artifacts/pilot30_baseline_md")
DEFAULT_ENHANCED_DIR = Path("/Users/ken/MM/Pipeline/cluster_match/artifacts/pilot30_enhanced_md")
DEFAULT_OUTPUT_DIR = Path("/Users/ken/MM/Pipeline/cluster_match/artifacts")
DEFAULT_MODEL = "google/gemini-3.1-pro-preview"
DEFAULT_TIMEOUT = 180


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.exists() or not config_path.is_file():
        return {}
    raw = config_path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    parsed = yaml.safe_load(raw)
    return parsed if isinstance(parsed, dict) else {}


def get_config_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def read_text_secret_file(path: str | Path | None, *, line_number: int = 1) -> str:
    if not path:
        return ""
    secret_path = Path(path).expanduser()
    if not secret_path.exists() or not secret_path.is_file():
        return ""
    try:
        lines = [line.strip() for line in secret_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return ""
    if not lines:
        return ""
    index = max(1, int(line_number or 1)) - 1
    if index >= len(lines):
        return ""
    return lines[index]


@dataclass(slots=True)
class RuntimeConfig:
    api_key: str
    base_url: str
    model: str
    timeout: int
    config_path: Path
    dataset_path: Path
    baseline_dir: Path
    enhanced_dir: Path
    output_dir: Path

    @classmethod
    def from_sources(
        cls,
        *,
        config_path: str | Path = DEFAULT_FINAL_CONFIG_PATH,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        baseline_dir: str | Path = DEFAULT_BASELINE_DIR,
        enhanced_dir: str | Path = DEFAULT_ENHANCED_DIR,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        model: str | None = None,
        timeout: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> "RuntimeConfig":
        config_path = Path(config_path).expanduser()
        config = load_yaml_config(config_path)

        key_env = str(get_config_value(config, "api", "key_env", default="OPENAI_API_KEY")).strip() or "OPENAI_API_KEY"
        key_file = str(get_config_value(config, "api", "key_file", default="")).strip()
        key_line = max(1, int(get_config_value(config, "api", "key_line", default=1) or 1))
        resolved_api_key = (
            api_key
            or os.getenv("CLUSTER_MATCH_API_KEY")
            or str(get_config_value(config, "api", "key", default="")).strip()
            or read_text_secret_file(key_file, line_number=key_line)
            or os.getenv(key_env)
            or os.getenv("OPENAI_API_KEY")
        )
        resolved_base_url = (
            base_url
            or os.getenv("CLUSTER_MATCH_BASE_URL")
            or str(get_config_value(config, "api", "base_url", default="")).strip()
        )
        resolved_model = (
            model
            or os.getenv("CLUSTER_MATCH_MODEL")
            or DEFAULT_MODEL
        )
        resolved_timeout = int(
            timeout
            or os.getenv("CLUSTER_MATCH_TIMEOUT")
            or get_config_value(config, "api", "timeout", default=DEFAULT_TIMEOUT)
            or DEFAULT_TIMEOUT
        )

        if not resolved_api_key:
            raise RuntimeError("API key is missing. Check final_version/config.yaml api.key/api.key_file or export OPENAI_API_KEY.")
        if not resolved_base_url:
            raise RuntimeError("Base URL is missing. Check final_version/config.yaml.")

        return cls(
            api_key=resolved_api_key,
            base_url=resolved_base_url.rstrip("/"),
            model=resolved_model,
            timeout=max(1, resolved_timeout),
            config_path=config_path,
            dataset_path=Path(dataset_path).expanduser(),
            baseline_dir=Path(baseline_dir).expanduser(),
            enhanced_dir=Path(enhanced_dir).expanduser(),
            output_dir=Path(output_dir).expanduser(),
        )
