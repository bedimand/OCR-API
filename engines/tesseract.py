import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pytesseract
from pytesseract import Output

from engines.utils import clamp_unit, compute_caps_ratio, load_pages


def format_page_lines(image, lang: str, psm: int, config: str | None = None) -> List[str]:
    width, height = image.size
    tess_config = config or ""
    if psm is not None:
        tess_config = f"{tess_config} --psm {psm}".strip()

    data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT, config=tess_config)
    line_groups: Dict[Tuple[int, int, int], Dict[str, List]] = defaultdict(lambda: {
        "texts": [],
        "xs": [],
        "ys": [],
        "xe": [],
        "ye": [],
    })

    n = len(data["text"])
    for idx in range(n):
        text = (data["text"][idx] or "").strip()
        if not text:
            continue
        left = data["left"][idx]
        top = data["top"][idx]
        width_box = data["width"][idx]
        height_box = data["height"][idx]
        block = data["block_num"][idx]
        paragraph = data["par_num"][idx]
        line = data["line_num"][idx]

        group = line_groups[(block, paragraph, line)]
        group["texts"].append(text)
        group["xs"].append(left)
        group["ys"].append(top)
        group["xe"].append(left + width_box)
        group["ye"].append(top + height_box)

    ordered_lines: List[Tuple[float, float, str]] = []
    for group in line_groups.values():
        line_text = " ".join(group["texts"]).strip()
        if not line_text:
            continue
        min_x = float(min(group["xs"]))
        min_y = float(min(group["ys"]))
        norm_x = clamp_unit(min_x / width)
        norm_y = clamp_unit(min_y / height)
        caps_ratio = compute_caps_ratio(line_text)
        formatted = f"[x:{norm_x:.2f}, y:{norm_y:.2f}, caps:{caps_ratio:.2f}] {line_text}"
        ordered_lines.append((norm_y, norm_x, formatted))

    ordered_lines.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in ordered_lines]


def run(input_path: Path, dpi: int, max_pages: int, lang: str, psm: int, config: str | None) -> int:
    if not input_path.is_file():
        print(f"File not found: {input_path}", file=sys.stderr)
        return 1
    try:
        pages, truncated = load_pages(input_path, dpi=dpi, max_pages=max_pages)
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    if not pages:
        print("No pages found in input", file=sys.stderr)
        return 1
    if truncated:
        print(
            f"Warning: PDF has more than {max_pages} pages; only processing the first {max_pages} pages",
            file=sys.stderr,
        )
    multi_page = len(pages) > 1
    for page_number, page in enumerate(pages, start=1):
        lines = format_page_lines(page, lang=lang, psm=psm, config=config)
        if multi_page:
            print(f"# Page {page_number}")
        for line in lines:
            print(line)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OCR text using Tesseract and emit formatted lines")
    parser.add_argument("input", type=Path, help="Path to the PDF or image file")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution used when rasterizing PDF pages")
    parser.add_argument("--max-pages", type=int, default=30, help="Maximum number of pages to process for PDFs")
    parser.add_argument("--lang", default="eng", help="Language code for Tesseract (e.g. eng, por)")
    parser.add_argument(
        "--psm", type=int, default=6, help="Tesseract page segmentation mode (see `tesseract --help-psm`)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Additional configuration string passed to Tesseract",
    )
    args = parser.parse_args()
    exit_code = run(args.input, args.dpi, args.max_pages, args.lang, args.psm, args.config)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
