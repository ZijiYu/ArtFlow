from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("缺少 PyYAML，无法生成消融实验配置。") from exc


REPO_ROOT = Path("/Users/ken/MM/Pipeline")
FINAL_VERSION_ROOT = REPO_ROOT / "final_version"
DEFAULT_CONFIG_PATH = FINAL_VERSION_ROOT / "config.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "ablation"
DEFAULT_ENTRYPOINT = FINAL_VERSION_ROOT / "pics" / "closed_loop.py"

MODULE_FLAG_MAP = {
    "round_table_validation": "disable_round_table_validation",
    "reflection_layer": "disable_reflection_layer",
    "preception_layer": "disable_preception_layer",
    "cot_layer": "disable_cot_layer",
}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def normalize_order(raw_order: object) -> list[str]:
    if not isinstance(raw_order, list):
        return list(MODULE_FLAG_MAP)
    normalized: list[str] = []
    for item in raw_order:
        key = str(item or "").strip()
        if key in MODULE_FLAG_MAP and key not in normalized:
            normalized.append(key)
    for key in MODULE_FLAG_MAP:
        if key not in normalized:
            normalized.append(key)
    return normalized


def set_nested(config: dict[str, Any], *keys: str, value: Any) -> None:
    current = config
    for key in keys[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[keys[-1]] = value


def variant_name(step_index: int, disabled_modules: list[str]) -> str:
    if step_index == 0:
        return "baseline"
    suffix = "_".join(f"no_{name}" for name in disabled_modules)
    return f"step_{step_index:02d}_{suffix}"


def build_variants(order: list[str]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = [
        {
            "name": "baseline",
            "disabled_modules": [],
            "flags": {flag_name: False for flag_name in MODULE_FLAG_MAP.values()},
        }
    ]
    cumulative: list[str] = []
    for index, module_name in enumerate(order, start=1):
        cumulative.append(module_name)
        flags = {flag_name: False for flag_name in MODULE_FLAG_MAP.values()}
        for disabled in cumulative:
            flags[MODULE_FLAG_MAP[disabled]] = True
        variants.append(
            {
                "name": variant_name(index, cumulative),
                "disabled_modules": list(cumulative),
                "flags": flags,
            }
        )
    return variants


def apply_variant_config(
    base_config: dict[str, Any],
    *,
    variant: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    variant_output_dir = output_root / variant["name"]
    ablation_section = config.setdefault("ablation", {})
    modules_section = ablation_section.setdefault("modules", {})
    ablation_section["enabled"] = True
    ablation_section["variant"] = variant["name"]
    ablation_section["output_root"] = str(output_root)
    for key, value in variant["flags"].items():
        modules_section[key] = bool(value)

    set_nested(config, "runtime", "output_dir", value=str(variant_output_dir))
    set_nested(config, "closed_loop", "output_dir", value=str(variant_output_dir))
    return config


def run_variant(config_path: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(DEFAULT_ENTRYPOINT),
        "--config",
        str(config_path),
    ]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate cumulative ablation configs for final_version.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="基础 config.yaml 路径")
    parser.add_argument("--output-root", default="", help="实验输出根目录，默认读取 ablation.output_root")
    parser.add_argument("--run", action="store_true", help="生成配置后立即执行所有变体")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser()
    base_config = load_yaml(config_path)
    ablation_config = base_config.get("ablation", {}) if isinstance(base_config.get("ablation"), dict) else {}
    output_root = Path(
        str(args.output_root).strip() or str(ablation_config.get("output_root", DEFAULT_OUTPUT_ROOT))
    ).expanduser()
    variants = build_variants(normalize_order(ablation_config.get("order")))

    output_root.mkdir(parents=True, exist_ok=True)
    configs_dir = output_root / "configs"
    summary: dict[str, Any] = {
        "base_config": str(config_path),
        "output_root": str(output_root),
        "strategy": str(ablation_config.get("strategy", "cumulative") or "cumulative"),
        "variants": [],
    }

    for variant in variants:
        variant_config = apply_variant_config(base_config, variant=variant, output_root=output_root)
        variant_config_path = configs_dir / f"{variant['name']}.yaml"
        dump_yaml(variant_config_path, variant_config)
        command = f"{sys.executable} {DEFAULT_ENTRYPOINT} --config {variant_config_path}"
        record: dict[str, Any] = {
            "name": variant["name"],
            "disabled_modules": variant["disabled_modules"],
            "config_path": str(variant_config_path),
            "output_dir": str(output_root / variant["name"]),
            "command": command,
        }
        if args.run:
            result = run_variant(variant_config_path)
            log_path = output_root / variant["name"] / "ablation_stdout.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                "\n".join(
                    [
                        f"returncode={result.returncode}",
                        "",
                        "[stdout]",
                        result.stdout,
                        "",
                        "[stderr]",
                        result.stderr,
                    ]
                ),
                encoding="utf-8",
            )
            record["returncode"] = result.returncode
            record["log_path"] = str(log_path)
        summary["variants"].append(record)

    summary_path = output_root / "ablation_manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== ablation plan ready ===")
    print(f"Manifest: {summary_path}")
    for item in summary["variants"]:
        print(f"- {item['name']}: {item['command']}")


if __name__ == "__main__":
    main()
