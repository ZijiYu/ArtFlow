#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps


DEFAULT_MAX_BYTES = 1_003_520
JPEG_QUALITIES = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15]
MIN_DIMENSION = 256
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy local images referenced by a JSONL file and compress them under a size limit.",
    )
    parser.add_argument(
        "--input-jsonl",
        default="/Users/ken/MM/Pipeline/cluster_match/gt_52_cleaned.jsonl",
        help="Input JSONL file. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/yzj/images_test",
        help="Directory to store processed images. Default: %(default)s",
    )
    parser.add_argument(
        "--image-key",
        default="image",
        help="JSON key that contains the local image path. Default: %(default)s",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Maximum output file size in bytes. Default: %(default)s",
    )
    parser.add_argument(
        "--source-prefix",
        default="/data/share/data/jpg",
        help="Original absolute path prefix to replace when --source-root is used. Default: %(default)s",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Local directory that mirrors the source images. Useful when JSONL paths are not directly accessible.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional JSONL path for per-image results. Defaults to <output-dir>/download_results.jsonl",
    )
    parser.add_argument(
        "--updated-jsonl-path",
        default=None,
        help="Optional JSONL path to write records with the image field replaced by the processed file path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    safe_chars = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("._") or "image"


def resolve_local_source(source: str, source_root: str | None, source_prefix: str | None) -> Path:
    direct_path = Path(source).expanduser()
    if direct_path.exists():
        return direct_path
    if source_root is None:
        return direct_path

    replacement_root = Path(source_root).expanduser()
    posix_source = PurePosixPath(source)
    candidates: list[Path] = []

    if source_prefix:
        prefix = PurePosixPath(source_prefix)
        try:
            relative = posix_source.relative_to(prefix)
            candidates.append(replacement_root.joinpath(*relative.parts))
        except ValueError:
            pass

    stripped_parts = posix_source.parts[1:] if posix_source.is_absolute() else posix_source.parts
    if stripped_parts:
        candidates.append(replacement_root.joinpath(*stripped_parts))
    candidates.append(replacement_root / posix_source.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else direct_path


def read_local_source_bytes(source: str, source_root: str | None, source_prefix: str | None) -> tuple[bytes, str]:
    local_path = resolve_local_source(source, source_root=source_root, source_prefix=source_prefix)
    if not local_path.exists():
        raise FileNotFoundError(f"Source image not found: {source} (resolved to {local_path})")
    return local_path.read_bytes(), str(local_path)


def make_output_stem(source: str, line_number: int) -> str:
    source_name = Path(source).name
    stem = Path(source_name).stem or f"image_{line_number:04d}"
    return sanitize_name(stem)


def output_suffix(source: str, used_original: bool) -> str:
    suffix = Path(source).suffix.lower()
    if used_original and suffix:
        return suffix
    return ".jpg"


def convert_for_jpeg(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGB", "L"}:
        return image.convert("RGB")
    if image.mode == "RGBA" or "A" in image.getbands():
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        background.alpha_composite(image.convert("RGBA"))
        return background.convert("RGB")
    return image.convert("RGB")


def encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    try:
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
        )
    except OSError:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def compress_image(raw_bytes: bytes, max_bytes: int) -> tuple[bytes, dict[str, Any]]:
    with Image.open(io.BytesIO(raw_bytes)) as original_image:
        prepared = convert_for_jpeg(original_image)
        base_width, base_height = prepared.size
        best_bytes: bytes | None = None
        best_meta: dict[str, Any] | None = None
        scale = 1.0

        while True:
            if scale == 1.0:
                working = prepared
            else:
                new_width = max(1, int(round(base_width * scale)))
                new_height = max(1, int(round(base_height * scale)))
                working = prepared.resize((new_width, new_height), RESAMPLE_LANCZOS)

            for quality in JPEG_QUALITIES:
                candidate_bytes = encode_jpeg(working, quality)
                candidate_meta = {
                    "quality": quality,
                    "width": working.width,
                    "height": working.height,
                    "scale": round(scale, 4),
                    "transcoded": True,
                }
                if best_bytes is None or len(candidate_bytes) < len(best_bytes):
                    best_bytes = candidate_bytes
                    best_meta = candidate_meta
                if len(candidate_bytes) <= max_bytes:
                    return candidate_bytes, candidate_meta

            if min(working.width, working.height) <= MIN_DIMENSION:
                break
            scale *= 0.9

        if best_bytes is None or best_meta is None:
            raise RuntimeError("Failed to compress image.")
        if len(best_bytes) > max_bytes:
            raise RuntimeError(
                f"Compressed image is still too large: {len(best_bytes)} bytes > limit {max_bytes} bytes"
            )
        return best_bytes, best_meta


def process_image(
    *,
    source: str,
    line_number: int,
    output_dir: Path,
    max_bytes: int,
    source_root: str | None,
    source_prefix: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    raw_bytes, resolved_source = read_local_source_bytes(
        source,
        source_root=source_root,
        source_prefix=source_prefix,
    )
    original_size = len(raw_bytes)
    used_original = original_size <= max_bytes
    final_bytes = raw_bytes
    compression_meta: dict[str, Any] = {
        "transcoded": False,
        "quality": None,
    }

    if not used_original:
        final_bytes, compression_meta = compress_image(raw_bytes, max_bytes=max_bytes)

    output_path = output_dir / f"{make_output_stem(source, line_number)}{output_suffix(source, used_original)}"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Use --overwrite to replace it.")

    output_path.write_bytes(final_bytes)

    return {
        "line_number": line_number,
        "source": source,
        "resolved_source": resolved_source,
        "output_path": str(output_path),
        "original_bytes": original_size,
        "final_bytes": len(final_bytes),
        "within_limit": len(final_bytes) <= max_bytes,
        "used_original": used_original,
        "compression": compression_meta,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    input_jsonl = Path(args.input_jsonl).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    manifest_path = Path(args.manifest_path).expanduser() if args.manifest_path else output_dir / "download_results.jsonl"
    updated_jsonl_path = Path(args.updated_jsonl_path).expanduser() if args.updated_jsonl_path else None

    if not input_jsonl.exists():
        print(f"Input JSONL not found: {input_jsonl}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if updated_jsonl_path is not None:
        updated_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    updated_records: list[dict[str, Any]] = []
    processed_cache: dict[str, dict[str, Any]] = {}
    success_count = 0

    with input_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue

            record = json.loads(text)
            source = record.get(args.image_key)
            if not isinstance(source, str) or not source.strip():
                skip_row = {
                    "line_number": line_number,
                    "status": "skipped",
                    "reason": f"Missing string field: {args.image_key}",
                }
                manifest_rows.append(skip_row)
                if updated_jsonl_path is not None:
                    updated_records.append(dict(record))
                continue
            source = source.strip()

            try:
                cached_result = processed_cache.get(source)
                if cached_result is None:
                    result = process_image(
                        source=source,
                        line_number=line_number,
                        output_dir=output_dir,
                        max_bytes=args.max_bytes,
                        source_root=args.source_root,
                        source_prefix=args.source_prefix,
                        overwrite=args.overwrite,
                    )
                    processed_cache[source] = dict(result)
                else:
                    result = dict(cached_result)
                    result["line_number"] = line_number

                result["status"] = "ok"
                manifest_rows.append(result)
                success_count += 1

                if updated_jsonl_path is not None:
                    updated_record = dict(record)
                    updated_record[args.image_key] = result["output_path"]
                    updated_records.append(updated_record)
            except Exception as exc:  # noqa: BLE001
                error_row = {
                    "line_number": line_number,
                    "source": source,
                    "status": "error",
                    "reason": str(exc),
                }
                manifest_rows.append(error_row)
                if updated_jsonl_path is not None:
                    updated_records.append(dict(record))

    write_jsonl(manifest_path, manifest_rows)
    if updated_jsonl_path is not None:
        write_jsonl(updated_jsonl_path, updated_records)

    error_count = len([row for row in manifest_rows if row.get("status") == "error"])
    skipped_count = len([row for row in manifest_rows if row.get("status") == "skipped"])

    print(f"Input JSONL: {input_jsonl}")
    print(f"Output directory: {output_dir}")
    print(f"Manifest: {manifest_path}")
    if updated_jsonl_path is not None:
        print(f"Updated JSONL: {updated_jsonl_path}")
    print(
        f"Processed {success_count} image(s), {error_count} error(s), {skipped_count} skipped."
    )

    return 0 if error_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
