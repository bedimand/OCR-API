from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import fitz  # PyMuPDF
from PIL import Image

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def compute_caps_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    uppercase = sum(1 for ch in letters if ch.isupper())
    return uppercase / len(letters)


def clamp_unit(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def render_pdf(path: Path, dpi: int) -> List[Image.Image]:
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    images: List[Image.Image] = []
    with fitz.open(path) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            mode = "RGB" if pix.n < 4 else "RGBA"
            image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            images.append(image.convert("RGB"))
    return images


def load_pages(input_path: Path, dpi: int, max_pages: int) -> Tuple[List[Image.Image], bool]:
    truncated = False
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        pages = render_pdf(input_path, dpi=dpi)
        if len(pages) > max_pages:
            truncated = True
            pages = pages[:max_pages]
        return pages, truncated
    if suffix in SUPPORTED_IMAGE_EXTS:
        with Image.open(str(input_path)) as image:
            return [image.convert("RGB")], truncated
    raise RuntimeError("Unsupported file type. Provide a PDF or an image (JPG/PNG/BMP/TIFF).")


def aggregate_entries(
    entries: Iterable[Dict[str, float]],
    width: int,
    height: int,
    vertical_factor: float = 0.6,
    vertical_bias: float = 3.0,
) -> List[str]:
    materialized = [entry for entry in entries if entry.get("text")]
    if not materialized:
        return []

    materialized.sort(key=lambda item: (item["center_y"], item["min_x"]))

    groups: List[Dict[str, object]] = []
    for entry in materialized:
        assigned = False
        for group in groups:
            dy = abs(entry["center_y"] - group["center_y"])
            threshold_y = max(group["avg_height"], entry["height"]) * vertical_factor + vertical_bias
            if dy <= threshold_y:
                items: List[Dict[str, float]] = group["items"]  # type: ignore[assignment]
                items.append(entry)
                count = len(items)
                group["center_y"] = ((group["center_y"] * (count - 1)) + entry["center_y"]) / count  # type: ignore[index]
                group["avg_height"] = ((group["avg_height"] * (count - 1)) + entry["height"]) / count  # type: ignore[index]
                group["min_x"] = min(group["min_x"], entry["min_x"])  # type: ignore[index]
                group["min_y"] = min(group["min_y"], entry["min_y"])  # type: ignore[index]
                assigned = True
                break
        if not assigned:
            groups.append(
                {
                    "center_y": entry["center_y"],
                    "avg_height": entry["height"],
                    "min_x": entry["min_x"],
                    "min_y": entry["min_y"],
                    "items": [entry],
                }
            )

    groups.sort(key=lambda group: group["center_y"])  # type: ignore[index]

    lines: List[Tuple[float, float, float, str]] = []
    for group in groups:
        items = sorted(group["items"], key=lambda item: item["min_x"])  # type: ignore[index]
        combined_text = " ".join(entry["text"] for entry in items).strip()
        if not combined_text:
            continue
        seg_min_x = min(entry["min_x"] for entry in items)
        seg_min_y = min(entry["min_y"] for entry in items)
        norm_x = clamp_unit(seg_min_x / width)
        norm_y = clamp_unit(seg_min_y / height)
        caps_ratio = compute_caps_ratio(combined_text)
        lines.append((norm_y, norm_x, caps_ratio, combined_text))

    lines.sort(key=lambda item: (item[0], item[1]))
    return [f"[x:{x:.2f}, y:{y:.2f}, caps:{caps:.2f}] {text}" for y, x, caps, text in lines]
