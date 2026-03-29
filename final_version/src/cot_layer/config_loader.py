from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("/Users/ken/MM/Pipeline/final_version/config.yaml")
DEFAULT_SLOTS_FILE = Path("/Users/ken/MM/Pipeline/final_version/preception_layer/artifacts/slots.jsonl")


def _import_yaml() -> Any:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        return None
    return yaml


@lru_cache(maxsize=8)
def load_yaml_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.exists() or not config_path.is_file():
        return {}

    raw = config_path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    yaml = _import_yaml()
    if yaml is None:
        return {}
    data = yaml.safe_load(raw)
    return data if isinstance(data, dict) else {}


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
