"""Benchmark OCR engines performance on the FUNSD dataset."""

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from difflib import SequenceMatcher
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from easyocr import Reader as EasyOCRReader
from paddleocr import PaddleOCR
from PIL import Image

from engines import easyocr, paddle, tesseract
from engines.utils import SUPPORTED_IMAGE_EXTS, aggregate_entries


@dataclass
class SampleResult:
    path: Path
    duration: float
    similarity: Optional[float]
    char_similarity: Optional[float]
    expected_chars: int
    predicted_chars: int
    lines: List[str]


@dataclass
class FunsdSample:
    image_path: Path
    expected_text: str


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


def downscale_image(image: Image.Image, factor: float) -> Image.Image:
    if factor is None or factor <= 0:
        return image
    if factor >= 0.9999:
        return image
    width, height = image.size
    new_width = max(1, int(width * factor))
    new_height = max(1, int(height * factor))
    if new_width == width and new_height == height:
        return image
    return image.resize((new_width, new_height))


def collect_funsd_samples(root: Path, split: str, limit: int) -> List[FunsdSample]:
    base = root / f"{split}_data"
    ann_dir = base / "annotations"
    img_dir = base / "images"
    if not ann_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(f"Invalid FUNSD structure under {base}")

    samples: List[FunsdSample] = []
    for ann_path in sorted(ann_dir.glob("*.json")):
        base_name = ann_path.stem
        image_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = img_dir / f"{base_name}{ext}"
            if candidate.exists():
                image_path = candidate
                break
        if image_path is None:
            print(f"Skipping {ann_path.name}: image not found", file=sys.stderr)
            continue

        try:
            expected_text = build_expected_text(image_path, ann_path)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load {ann_path.name}: {exc}", file=sys.stderr)
            continue

        samples.append(FunsdSample(image_path=image_path, expected_text=expected_text))
        if limit and limit > 0 and len(samples) >= limit:
            break
    return samples


def build_expected_text(image_path: Path, annotation_path: Path) -> str:
    with Image.open(image_path) as img:
        rgb_image = img.convert("RGB")
        width, height = rgb_image.size
    with annotation_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    entries: List[Dict[str, float]] = []
    for field in payload.get("form", []):
        for word in field.get("words", []):
            text = (word.get("text") or "").strip()
            if not text:
                continue
            box = word.get("box") or []
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            min_x = float(min(x1, x2))
            max_x = float(max(x1, x2))
            min_y = float(min(y1, y2))
            max_y = float(max(y1, y2))
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

    lines = aggregate_entries(entries, width, height)
    return strip_formatting(lines)


def benchmark_engine(
    engine: str,
    samples: Sequence[FunsdSample],
    downscale_factor: float,
    paddle_lang: str,
    tess_lang: str,
    tess_psm: int,
    tess_config: Optional[str],
    easyocr_langs: List[str],
    easyocr_gpu: bool,
) -> List[SampleResult]:
    if engine == "paddle":
        ocr = PaddleOCR(use_textline_orientation=True, lang=paddle_lang)

        def process_page(page: Image.Image) -> List[str]:
            return paddle.format_page_lines(page, ocr)

    elif engine == "tesseract":

        def process_page(page: Image.Image) -> List[str]:
            return tesseract.format_page_lines(page, lang=tess_lang, psm=tess_psm, config=tess_config)

    elif engine == "easyocr":
        reader = EasyOCRReader(easyocr_langs, gpu=easyocr_gpu)

        def process_page(page: Image.Image) -> List[str]:
            return easyocr.format_page_lines(page, reader)

    else:
        raise ValueError(f"Unsupported engine: {engine}")

    results: List[SampleResult] = []
    for sample in samples:
        try:
            with Image.open(sample.image_path) as img:
                base_image = img.convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {sample.image_path.name}: {exc}", file=sys.stderr)
            continue

        image_for_ocr = downscale_image(base_image, downscale_factor)

        start = time.perf_counter()
        try:
            lines = process_page(image_for_ocr)
        except Exception as exc:  # noqa: BLE001
            print(f"[{engine}] OCR failed on {sample.image_path.name}: {exc}", file=sys.stderr)
            continue
        end = time.perf_counter()

        flattened_text = strip_formatting(lines)
        seq_similarity, char_similarity, expected_chars, predicted_chars = compute_similarity(
            sample.expected_text,
            flattened_text,
        )

        results.append(
            SampleResult(
                path=sample.image_path,
                duration=end - start,
                similarity=seq_similarity,
                char_similarity=char_similarity,
                expected_chars=expected_chars,
                predicted_chars=predicted_chars,
                lines=lines,
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
    parser = argparse.ArgumentParser(description="Benchmark OCR engines on the FUNSD dataset")
    parser.add_argument("--root", type=Path, default=Path("dataset-funsd"), help="FUNSD root directory")
    parser.add_argument("--split", choices=["training", "testing"], default="testing", help="Dataset split to benchmark")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of documents (0 means all)")
    parser.add_argument("--downscale", type=float, default=1.0, help="Uniform scaling factor applied before OCR (<=1 to downscale)")
    parser.add_argument("--paddle-lang", default="en", help="Language code for PaddleOCR")
    parser.add_argument("--tesseract-lang", default="eng", help="Language code for Tesseract")
    parser.add_argument("--tesseract-psm", type=int, default=6, help="Tesseract page segmentation mode")
    parser.add_argument("--tesseract-config", default=None, help="Additional config string passed to Tesseract")
    parser.add_argument("--easyocr-langs", default="en", help="Comma-separated language codes for EasyOCR")
    parser.add_argument("--easyocr-gpu", action="store_true", help="Enable GPU acceleration for EasyOCR")
    parser.add_argument("--export-results", action="store_true", help="Write OCR outputs to the results/ directory")
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["paddle", "tesseract", "easyocr"],
        choices=["paddle", "tesseract", "easyocr"],
        help="OCR engines to benchmark",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.exists():
        print(f"FUNSD root not found: {root}", file=sys.stderr)
        sys.exit(1)

    samples = collect_funsd_samples(root, args.split, args.limit)
    if not samples:
        print("No FUNSD samples found. Check --root and --split values.")
        sys.exit(1)

    easyocr_langs = [code.strip() for code in args.easyocr_langs.split(",") if code.strip()]
    if not easyocr_langs:
        easyocr_langs = ["en"]

    print(
        f"Benchmarking {len(samples)} document(s) from {args.split} split under {root} "
        f"(downscale={args.downscale})"
    )

    summaries: List[dict] = []
    all_results: dict[str, List[SampleResult]] = {}
    for engine in args.engines:
        results = benchmark_engine(
            engine=engine,
            samples=samples,
            downscale_factor=args.downscale,
            paddle_lang=args.paddle_lang,
            tess_lang=args.tesseract_lang,
            tess_psm=args.tesseract_psm,
            tess_config=args.tesseract_config,
            easyocr_langs=easyocr_langs,
            easyocr_gpu=args.easyocr_gpu,
        )
        all_results[engine] = results
        summary = summarize(engine, results)
        if summary:
            summaries.append(summary)

    if summaries:
        output_prefix = f"benchmark_funsd_{args.split}"
        save_plots(summaries, output_prefix)

    if args.export_results:
        export_dir = Path("results")
        for engine, results in all_results.items():
            export_results(engine, results, dataset_root=root, output_dir=export_dir)


if __name__ == "__main__":
    main()

