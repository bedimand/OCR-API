"""Benchmark OCR engines performance across the dataset."""

import argparse
import csv
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from difflib import SequenceMatcher
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from easyocr import Reader as EasyOCRReader
from paddleocr import PaddleOCR

from engines import easyocr, paddle, tesseract
from engines.utils import SUPPORTED_IMAGE_EXTS, load_pages


@dataclass
class SampleResult:
    path: Path
    duration: float
    similarity: Optional[float]
    char_similarity: Optional[float]
    expected_chars: int
    predicted_chars: int
    lines: List[str]


def read_ground_truth(csv_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = (row.get("File Name") or "").strip()
            ocr_text = (row.get("OCRed Text") or "").strip()
            if not file_name:
                continue
            mapping[file_name] = ocr_text
    return mapping


def iter_labeled_images(root: Path) -> Iterable[Tuple[Path, Optional[str]]]:
    for csv_path in sorted(root.rglob("*.csv")):
        gt = read_ground_truth(csv_path)
        image_dir = csv_path.parent / csv_path.stem
        for file_name, expected_text in gt.items():
            image_path = image_dir / file_name
            yield image_path, expected_text


def iter_unlabeled_images(root: Path) -> Iterable[Tuple[Path, None]]:
    valid_exts = SUPPORTED_IMAGE_EXTS | {".pdf"}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in valid_exts:
            yield path, None


def flatten_lines(pages: Sequence[Sequence[str]]) -> List[str]:
    flat: List[str] = []
    for page_index, lines in enumerate(pages, start=1):
        if len(pages) > 1:
            flat.append(f"# Page {page_index}")
        flat.extend(lines)
    return flat


def strip_formatting(lines: Sequence[str]) -> str:
    texts: List[str] = []
    for line in lines:
        if line.startswith("# Page"):
            continue
        if "] " in line:
            _, text = line.split("] ", 1)
        else:
            text = line
        texts.append(text)
    return "\n".join(texts)


def normalize_for_similarity(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.lower()
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_for_char_accuracy(value: str) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.lower()
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9]+", "", normalized)
    return normalized


def compute_similarity(expected: str, predicted: str) -> Tuple[Optional[float], Optional[float], int, int]:
    normalized_exp = normalize_for_similarity(expected)
    normalized_pred = normalize_for_similarity(predicted)
    char_exp = normalize_for_char_accuracy(expected)
    char_pred = normalize_for_char_accuracy(predicted)

    seq_similarity = None
    char_similarity = None

    if normalized_exp and normalized_pred:
        seq_similarity = SequenceMatcher(None, normalized_exp, normalized_pred).ratio()
    elif not normalized_exp and not normalized_pred:
        seq_similarity = 1.0

    if char_exp and char_pred:
        char_similarity = SequenceMatcher(None, char_exp, char_pred).ratio()
    elif not char_exp and not char_pred:
        char_similarity = 1.0

    return seq_similarity, char_similarity, len(char_exp), len(char_pred)


def downscale_pages(pages: Sequence, factor: float):
    if factor is None or factor <= 0:
        return list(pages)
    if factor >= 0.9999:
        return list(pages)
    resized = []
    for page in pages:
        width, height = page.size
        new_width = max(1, int(width * factor))
        new_height = max(1, int(height * factor))
        if new_width == width and new_height == height:
            resized.append(page)
        else:
            resized.append(page.resize((new_width, new_height)))
    return resized


def benchmark_engine(
    engine: str,
    samples: Sequence[Tuple[Path, Optional[str]]],
    dpi: int,
    max_pages: int,
    paddle_lang: str,
    tess_lang: str,
    tess_psm: int,
    tess_config: Optional[str],
    easyocr_langs: List[str],
    easyocr_gpu: bool,
    downscale_factor: float,
) -> List[SampleResult]:
    if engine == "paddle":
        ocr = PaddleOCR(use_textline_orientation=True, lang=paddle_lang)

        def process_page(page):
            return paddle.format_page_lines(page, ocr)

    elif engine == "tesseract":

        def process_page(page):
            return tesseract.format_page_lines(page, lang=tess_lang, psm=tess_psm, config=tess_config)

    elif engine == "easyocr":
        reader = EasyOCRReader(easyocr_langs, gpu=easyocr_gpu)

        def process_page(page):
            return easyocr.format_page_lines(page, reader)

    else:
        raise ValueError(f"Unsupported engine: {engine}")

    results: List[SampleResult] = []
    for path, expected_text in samples:
        if not path.exists():
            print(f"Skipping missing file: {path}", file=sys.stderr)
            continue
        try:
            pages, truncated = load_pages(path, dpi=dpi, max_pages=max_pages)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load {path}: {exc}", file=sys.stderr)
            continue
        if not pages:
            print(f"No pages found in {path}", file=sys.stderr)
            continue
        if truncated:
            print(
                f"[{engine}] Warning: {path} exceeds {max_pages} pages, only first {max_pages} processed",
                file=sys.stderr,
            )

        pages = downscale_pages(pages, downscale_factor)

        start = time.perf_counter()
        page_lines: List[List[str]] = []
        for page in pages:
            page_lines.append(process_page(page))
        end = time.perf_counter()

        flat_lines = flatten_lines(page_lines)
        flattened_text = strip_formatting(flat_lines)
        seq_similarity = None
        char_similarity = None
        expected_chars = predicted_chars = 0
        if expected_text is not None:
            seq_similarity, char_similarity, expected_chars, predicted_chars = compute_similarity(
                expected_text,
                flattened_text,
            )
        else:
            predicted_chars = len(flattened_text)

        results.append(
            SampleResult(
                path=path,
                duration=end - start,
                similarity=seq_similarity,
                char_similarity=char_similarity,
                expected_chars=expected_chars,
                predicted_chars=predicted_chars,
                lines=flat_lines,
            )
        )
    return results


def summarize(engine: str, results: Sequence[SampleResult]):
    if not results:
        print(f"No samples processed for engine {engine}.")
        return None
    durations = [r.duration for r in results]
    total_time = sum(durations)
    avg_time = statistics.mean(durations)
    median_time = statistics.median(durations)
    p95_time = statistics.quantiles(durations, n=100)[94] if len(durations) >= 100 else None

    print(f"\n=== Engine: {engine} ===")
    print(f"Processed {len(results)} documents")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per doc: {avg_time:.2f}s")
    print(f"Median time per doc: {median_time:.2f}s")
    if p95_time is not None:
        print(f"95th percentile time: {p95_time:.2f}s")

    seq_similarities = [r.similarity for r in results if r.similarity is not None]
    avg_token = statistics.mean(seq_similarities) if seq_similarities else None
    median_token = statistics.median(seq_similarities) if seq_similarities else None
    if seq_similarities:
        print(f"Average similarity (token): {avg_token:.3f}")
        print(f"Median similarity (token): {median_token:.3f}")
    else:
        print("Average similarity (token): N/A")

    char_similarities = [r.char_similarity for r in results if r.char_similarity is not None]
    avg_char = statistics.mean(char_similarities) if char_similarities else None
    median_char = statistics.median(char_similarities) if char_similarities else None
    if char_similarities:
        print(f"Average similarity (char): {avg_char:.3f}")
        print(f"Median similarity (char): {median_char:.3f}")
    else:
        print("Average similarity (char): N/A")

    return {
        "engine": engine,
        "documents": len(results),
        "total_time": total_time,
        "avg_time": avg_time,
        "median_time": median_time,
        "avg_token": avg_token,
        "median_token": median_token,
        "avg_char": avg_char,
        "median_char": median_char,
    }


def save_plots(summaries: List[dict], output_prefix: str) -> None:
    if not summaries:
        return
    engines = [s["engine"] for s in summaries]

    avg_times = [s["avg_time"] for s in summaries]
    plt.figure(figsize=(8, 4))
    plt.bar(engines, avg_times, color="steelblue")
    plt.ylabel("Average Time (s)")
    plt.title("Average Processing Time per Engine")
    plt.tight_layout()
    time_path = Path(f"{output_prefix}_time.png")
    plt.savefig(time_path)
    plt.close()

    if any(s["avg_token"] is not None for s in summaries):
        token_values = [s["avg_token"] if s["avg_token"] is not None else 0 for s in summaries]
        plt.figure(figsize=(8, 4))
        plt.bar(engines, token_values, color="seagreen")
        plt.ylabel("Average Token Similarity")
        plt.ylim(0, 1)
        plt.title("Average Token Similarity per Engine")
        plt.tight_layout()
        token_path = Path(f"{output_prefix}_token.png")
        plt.savefig(token_path)
        plt.close()
    else:
        token_path = None

    if any(s["avg_char"] is not None for s in summaries):
        char_values = [s["avg_char"] if s["avg_char"] is not None else 0 for s in summaries]
        plt.figure(figsize=(8, 4))
        plt.bar(engines, char_values, color="mediumpurple")
        plt.ylabel("Average Char Similarity")
        plt.ylim(0, 1)
        plt.title("Average Character Similarity per Engine")
        plt.tight_layout()
        char_path = Path(f"{output_prefix}_char.png")
        plt.savefig(char_path)
        plt.close()
    else:
        char_path = None

    print(f"Saved visualization charts: {time_path}", file=sys.stderr)
    if token_path:
        print(f"Saved token similarity chart: {token_path}", file=sys.stderr)
    if char_path:
        print(f"Saved character similarity chart: {char_path}", file=sys.stderr)


def export_results(engine: str, results: Sequence[SampleResult], dataset_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    engine_dir = output_dir / engine.replace(" ", "_")
    engine_dir.mkdir(exist_ok=True)
    for item in results:
        rel_name = item.path.name
        file_path = engine_dir / f"{Path(rel_name).stem}_{engine}.txt"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("\n".join(item.lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OCR engines across the dataset")
    parser.add_argument("--root", type=Path, default=Path("dataset-high-quality"), help="Dataset root directory to scan for files")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of documents to process (0 means all)")
    parser.add_argument("--dpi", type=int, default=300, help="DPI used when rasterizing PDFs")
    parser.add_argument("--max-pages", type=int, default=30, help="Maximum number of pages per PDF")
    parser.add_argument("--downscale", type=float, default=1.0, help="Uniform scaling factor applied before OCR (<=1 to downscale)")
    parser.add_argument("--paddle-lang", default="en", help="Language code for PaddleOCR")
    parser.add_argument("--tesseract-lang", default="eng", help="Language code for Tesseract")
    parser.add_argument("--tesseract-psm", type=int, default=6, help="Tesseract page segmentation mode")
    parser.add_argument("--tesseract-config", default=None, help="Additional config string passed to Tesseract")
    parser.add_argument("--easyocr-langs", default="en", help="Comma-separated language codes for EasyOCR")
    parser.add_argument("--easyocr-gpu", action="store_true", help="Enable GPU acceleration for EasyOCR")
    parser.add_argument("--include-unlabeled", action="store_true", help="Also benchmark unlabeled files (without similarity scores)")
    parser.add_argument("--export-results", action="store_true", help="Write OCR outputs to the results/ directory")
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["paddle"],
        choices=["paddle", "tesseract", "easyocr"],
        help="OCR engines to benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.exists():
        print(f"Root path not found: {root}", file=sys.stderr)
        sys.exit(1)

    labeled_samples = list(iter_labeled_images(root))
    samples: List[Tuple[Path, Optional[str]]] = labeled_samples

    if args.include_unlabeled:
        labeled_paths = {path for path, _ in labeled_samples}
        unlabeled = [item for item in iter_unlabeled_images(root) if item[0] not in labeled_paths]
        samples.extend(unlabeled)

    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    if not samples:
        print("No samples found. Adjust --root or --include-unlabeled.")
        sys.exit(1)

    easyocr_langs = [code.strip() for code in args.easyocr_langs.split(",") if code.strip()]
    if not easyocr_langs:
        easyocr_langs = ["en"]

    print(
        f"Benchmarking {len(samples)} document(s) from {root} "
        f"(downscale={args.downscale})"
    )

    summaries: List[dict] = []
    all_results: dict[str, List[SampleResult]] = {}
    for engine in args.engines:
        results = benchmark_engine(
            engine=engine,
            samples=samples,
            dpi=args.dpi,
            max_pages=args.max_pages,
            paddle_lang=args.paddle_lang,
            tess_lang=args.tesseract_lang,
            tess_psm=args.tesseract_psm,
            tess_config=args.tesseract_config,
            easyocr_langs=easyocr_langs,
            easyocr_gpu=args.easyocr_gpu,
            downscale_factor=args.downscale,
        )
        all_results[engine] = results
        summary = summarize(engine, results)
        if summary:
            summaries.append(summary)

    if summaries:
        save_plots(summaries, output_prefix="benchmark_ocr")

    if args.export_results:
        export_dir = Path("results")
        for engine, results in all_results.items():
            export_results(engine, results, dataset_root=root, output_dir=export_dir)


if __name__ == "__main__":
    main()

