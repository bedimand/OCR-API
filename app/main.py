from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from threading import Lock
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from paddleocr import PaddleOCR

from engines.paddle import format_page_lines
from engines.utils import load_pages

OCR_DPI = 300
OCR_MAX_PAGES = 30
OCR_LANG = "en"

_app_lock = Lock()
_paddle_ocr: PaddleOCR | None = None


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

    engine = get_paddle_engine()
    multi_page = len(pages) > 1
    lines: List[str] = []
    for index, page in enumerate(pages, start=1):
        page_lines = format_page_lines(page, engine)
        if multi_page:
            lines.append(f"# Page {index}")
        lines.extend(page_lines)
    return lines


app = FastAPI(title="OCR API", version="0.1.0")


@app.on_event("startup")
async def startup_event() -> None:
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

