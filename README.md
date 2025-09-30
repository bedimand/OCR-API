# OCR API

REST API built with FastAPI that accepts scanned PDFs (up to 30 pages), performs on-device OCR with PaddleOCR, and returns the extracted text line-by-line in the format `[x:�, y:�, caps:�] text`.

## Requirements & Setup

1. Install dependencies using [uv](https://docs.astral.sh/uv/):
   ```bash
   uv sync
   ```
2. Activate the virtual environment or prefix commands with `uv run`.

## Running the API

```bash
uv run uvicorn app.main:app --reload
```

The service exposes a single endpoint:

- `POST /upload` � multipart/form-data with a `file` field containing a PDF (=30 pages).

Example request:

```bash
curl -X POST \
  -F "file=@data/batch_1/batch_1/batch1_1/batch1-0494.pdf" \
  http://localhost:8000/upload
```

The response is `text/plain`, each line containing normalized coordinates and uppercase ratio, e.g.

```
[x:0.12, y:0.08, caps:0.86] NOME DO CLIENTE: JOAO DA SILVA
```

> **Note:** OCR runs on CPU PaddleOCR models and may take ~20?s for large PDFs; consider downscaling pages or deploying with GPU for lower latency.

