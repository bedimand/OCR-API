import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from easyocr import Reader

from engines.utils import aggregate_entries, load_pages


def extract_lines(results, width: int, height: int) -> List[str]:
    entries = []
    for item in results:
        if len(item) < 2:
            continue
        box, text = item[0], item[1]
        text = (text or "").strip()
        if not text:
            continue
        polygon = np.asarray(box, dtype=float)
        if polygon.size == 0:
            continue
        min_x = float(np.min(polygon[:, 0]))
        max_x = float(np.max(polygon[:, 0]))
        min_y = float(np.min(polygon[:, 1]))
        max_y = float(np.max(polygon[:, 1]))
        entries.append(
            {
                "text": text,
                "min_x": min_x,
                "max_x": max_x,
                "min_y": min_y,
                "max_y": max_y,
                "center_y": (min_y + max_y) / 2.0,
                "height": max_y - min_y,
            }
        )
    return aggregate_entries(entries, width, height)


def format_page_lines(image, reader: Reader) -> List[str]:
    width, height = image.size
    np_image = np.array(image)
    results = reader.readtext(np_image, detail=1, paragraph=False)
    return extract_lines(results, width, height)


def run(input_path: Path, dpi: int, max_pages: int, langs: List[str], gpu: bool) -> int:
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
    reader = Reader(langs, gpu=gpu)
    multi_page = len(pages) > 1
    for page_number, page in enumerate(pages, start=1):
        lines = format_page_lines(page, reader)
        if multi_page:
            print(f"# Page {page_number}")
        for line in lines:
            print(line)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract OCR text using EasyOCR and emit formatted lines")
    parser.add_argument("input", type=Path, help="Path to the PDF or image file")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution used when rasterizing PDF pages")
    parser.add_argument("--max-pages", type=int, default=30, help="Maximum number of pages to process for PDFs")
    parser.add_argument(
        "--langs",
        default="en",
        help="Comma-separated language codes for EasyOCR (e.g. en,pt)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU acceleration if available",
    )
    args = parser.parse_args()
    lang_list = [code.strip() for code in args.langs.split(",") if code.strip()]
    if not lang_list:
        lang_list = ["en"]
    exit_code = run(args.input, args.dpi, args.max_pages, lang_list, args.gpu)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
