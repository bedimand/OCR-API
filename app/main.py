from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from threading import Lock
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from paddleocr import PaddleOCR

from engines.paddle import format_page_lines as paddle_format_page_lines
from engines.utils import load_pages

OCR_DPI = 300
OCR_MAX_PAGES = 30
OCR_LANG = "en"
TESSERACT_PSM = 6
SUPPORTED_ENGINES = {"paddle", "tesseract"}
OCR_ENGINE = "paddle"

_app_lock = Lock()
_paddle_ocr: PaddleOCR | None = None
_ocr_lock = Lock()


_tesseract_formatter = None


def _get_tesseract_formatter():
    global _tesseract_formatter
    if _tesseract_formatter is None:
        try:
            from engines.tesseract import format_page_lines as tesseract_format_page_lines
        except ModuleNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Tesseract OCR is not available. Install the Tesseract binary and the pytesseract package, then restart the service."
                ),
            ) from exc
        _tesseract_formatter = tesseract_format_page_lines
    return _tesseract_formatter


_TESSERACT_ALIASES = {
    "en": "eng",
}


def _tesseract_lang() -> str:
    return _TESSERACT_ALIASES.get(OCR_LANG, OCR_LANG)


def get_paddle_engine() -> PaddleOCR:
    global _paddle_ocr
    if _paddle_ocr is None:
        with _app_lock:
            if _paddle_ocr is None:
                _paddle_ocr = PaddleOCR(use_textline_orientation=True, lang=OCR_LANG)
    return _paddle_ocr


def run_ocr(pdf_path: Path) -> List[str]:
    pages, truncated = load_pages(pdf_path, dpi=OCR_DPI, max_pages=OCR_MAX_PAGES)
    if truncated:
        raise HTTPException(status_code=400, detail=f"PDF exceeds {OCR_MAX_PAGES} pages limit")
    if not pages:
        raise HTTPException(status_code=400, detail="No pages found in PDF")

    if OCR_ENGINE not in SUPPORTED_ENGINES:
        raise HTTPException(status_code=500, detail=f"Configured OCR engine \"{OCR_ENGINE}\" is not supported")

    with _ocr_lock:
        if OCR_ENGINE == "paddle":
            engine = get_paddle_engine()
            return _run_paddle_pages(engine, pages)
        return _run_tesseract_pages(pages)


app = FastAPI(title="OCR API", version="0.1.0")


@app.on_event("startup")
async def startup_event() -> None:
    if OCR_ENGINE == "paddle":
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, get_paddle_engine)


@app.post("/upload", response_class=PlainTextResponse)
async def upload(file: UploadFile = File(...)) -> PlainTextResponse:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file provided")

    loop = asyncio.get_running_loop()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        lines = await loop.run_in_executor(None, run_ocr, tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return PlainTextResponse("\n".join(lines))



def _run_paddle_pages(engine: PaddleOCR, pages: List) -> List[str]:
    multi_page = len(pages) > 1
    lines: List[str] = []
    for index, page in enumerate(pages, start=1):
        page_lines = paddle_format_page_lines(page, engine)
        if multi_page:
            lines.append(f"# Page {index}")
        lines.extend(page_lines)
    return lines



def _run_tesseract_pages(pages: List) -> List[str]:
    multi_page = len(pages) > 1
    lines: List[str] = []
    tess_lang = _tesseract_lang()
    formatter = _get_tesseract_formatter()
    for index, page in enumerate(pages, start=1):
        page_lines = formatter(
            page,
            lang=tess_lang,
            psm=TESSERACT_PSM,
        )
        if multi_page:
            lines.append(f"# Page {index}")
        lines.extend(page_lines)
    return lines
