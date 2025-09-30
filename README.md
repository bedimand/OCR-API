# API OCR

API REST constru�da com FastAPI que recebe PDFs digitalizados (at� 30 p�ginas), executa OCR local com PaddleOCR e devolve o texto extra�do linha a linha no formato `[x:<valor>, y:<valor>, caps:<valor>] texto`.

## Requisitos e instala��o

1. Instale as depend�ncias com [uv](https://docs.astral.sh/uv/):
   ```bash
   uv sync
   ```
2. Ative o ambiente virtual (exemplo PowerShell) ou prefixe os comandos com `uv run`:
   ```powershell
   & .venv\Scripts\Activate.ps1
   ```

## Executando a API

```bash
uv run uvicorn app.main:app --reload
```

### Endpoint dispon�vel
- `POST /upload` � recebe o PDF no campo `file` do formul�rio `multipart/form-data`.

Exemplo de requisi��o (PowerShell + `curl.exe`):
```powershell
curl.exe -X POST http://127.0.0.1:8000/upload ^
  -F "file=@sample-invoice.pdf;type=application/pdf"
```

A resposta � `text/plain`, cada linha contendo coordenadas normalizadas e a raz�o de letras mai�sculas, por exemplo:
```
[x:0.12, y:0.08, caps:0.86] NOME DO CLIENTE: JOAO DA SILVA
```

> **Observa��o:** PaddleOCR roda no CPU por padr�o e pode levar ~10�20 s por documento. Para reduzir a lat�ncia, considere diminuir o DPI, aplicar downscale ou usar builds com GPU.

## Benchmarks

Disponibilizamos dois scripts:
- `scripts/benchmark_ocr.py` � avalia o conjunto `dataset-high-quality/`.
- `scripts/benchmark_funsd.py` � avalia o conjunto `dataset-funsd/`.

Ambos aceitam:
- `--engines` (`paddle`, `tesseract`, `easyocr`).
- `--downscale` para redimensionar p�ginas antes do OCR.
- `--export-results` para salvar as sa�das em `results/<engine>/<arquivo>_<engine>.txt`.
- Gera��o autom�tica de gr�ficos (`matplotlib`) salvos em `assets/`.

Exemplo (FUNSD):
```bash
uv run python scripts/benchmark_funsd.py \
  --engines paddle tesseract easyocr \
  --limit 50 \
  --export-results
```

## Resultados recentes (FUNSD)

Gr�ficos: `assets/benchmark_funsd_testing_time.png`, `assets/benchmark_funsd_testing_token.png`, `assets/benchmark_funsd_testing_char.png`.

| Engine     | Tempo m�dio (s) | Similaridade (token) | Similaridade (caracteres) |
|------------|-----------------|-----------------------|----------------------------|
| **Paddle** | 11,59           | **0,771**             | **0,783**                  |
| Tesseract  | 0,53            | 0,410                 | 0,413                      |
| EasyOCR    | 11,98           | 0,379                 | 0,362                      |

**Insights**
- *Precis�o*: PaddleOCR gera transcri��es mais fi�is, mantendo m�dia de ~0,78 nos caracteres.
- *Velocidade*: Tesseract � muito r�pido (~0,5 s/doc) por�m com perda acentuada de acur�cia.
- *EasyOCR*: fica no meio-termo, com tempo pr�ximo ao Paddle, mas menos preciso.
- *Trade-off*: utilize PaddleOCR quando qualidade � prioridade; Tesseract serve para extra��es r�pidas de baixa exig�ncia.

Os arquivos exportados seguem o padr�o `nomeoriginal_engine.txt` no diret�rio `results/`, facilitando auditorias e compara��es.
