from __future__ import annotations

import math
import tempfile
from pathlib import Path

from .models import PreparedImage


def prepare_image(image_path: str, max_pixel: int, resize: bool = True) -> PreparedImage:
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return PreparedImage(path=image_path, note="image_not_found_or_remote")

    try:
        from PIL import Image
    except ModuleNotFoundError:
        return PreparedImage(path=image_path, note="pillow_missing_skip_resize")

    original_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(path) as image:
            width, height = image.size
            original_pixels = width * height
            if not resize or original_pixels <= max_pixel:
                return PreparedImage(
                    path=image_path,
                    original_size=(width, height),
                    prepared_size=(width, height),
                    original_pixels=original_pixels,
                    prepared_pixels=original_pixels,
                    note="within_pixel_limit",
                )

            scale = math.sqrt(max_pixel / float(original_pixels))
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            resized = image.resize((new_width, new_height))
            suffix = path.suffix or ".png"
            with tempfile.NamedTemporaryFile(prefix="slots_v2_", suffix=suffix, delete=False, dir="/tmp") as handle:
                temp_path = handle.name
            resized.save(temp_path)
            return PreparedImage(
                path=temp_path,
                original_size=(width, height),
                prepared_size=(new_width, new_height),
                original_pixels=original_pixels,
                prepared_pixels=new_width * new_height,
                was_resized=True,
                note="resized_to_max_pixel",
            )
    finally:
        Image.MAX_IMAGE_PIXELS = original_limit
