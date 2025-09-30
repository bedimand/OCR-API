# API OCR

API REST construída com FastAPI que recebe PDFs digitalizados (até 30 páginas), executa OCR local com PaddleOCR e devolve o texto extraído linha a linha no formato `[x:<valor>, y:<valor>, caps:<valor>] texto`.

## Requisitos e instalação

1. Instale as dependências com [uv](https://docs.astral.sh/uv/):
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

### Endpoint disponível
- `POST /upload` – recebe o PDF no campo `file` do formulário `multipart/form-data`.

Exemplo de requisição (PowerShell + `curl.exe`):
```powershell
curl.exe -X POST http://127.0.0.1:8000/upload ^
  -F "file=@sample-invoice.pdf;type=application/pdf"
```

A resposta é `text/plain`, cada linha contendo coordenadas normalizadas e a razão de letras maiúsculas, por exemplo:
```
[x:0.12, y:0.08, caps:0.86] NOME DO CLIENTE: JOAO DA SILVA
```

> **Observação:** PaddleOCR roda no CPU por padrão e pode levar ~10–20 s por documento. Para reduzir a latência, considere diminuir o DPI, aplicar downscale ou usar builds com GPU.

## Benchmarks

Disponibilizamos dois scripts:
- `scripts/benchmark_ocr.py` – avalia o conjunto `dataset-high-quality/`.
- `scripts/benchmark_funsd.py` – avalia o conjunto `dataset-funsd/`.

Ambos aceitam:
- `--engines` (`paddle`, `tesseract`, `easyocr`).
- `--downscale` para redimensionar páginas antes do OCR.
- `--export-results` para salvar as saídas em `results/<engine>/<arquivo>_<engine>.txt`.
- Geração automática de gráficos (`matplotlib`) salvos em `assets/`.

Exemplo (FUNSD):
```bash
uv run python scripts/benchmark_funsd.py \
  --engines paddle tesseract easyocr \
  --limit 50 \
  --export-results
```

## Resultados recentes (FUNSD)

Gráficos: `assets/benchmark_funsd_testing_time.png`, `assets/benchmark_funsd_testing_token.png`, `assets/benchmark_funsd_testing_char.png`.

| Engine     | Tempo médio (s) | Similaridade (token) | Similaridade (caracteres) |
|------------|-----------------|-----------------------|----------------------------|
| **Paddle** | 11,59           | **0,771**             | **0,783**                  |
| Tesseract  | 0,53            | 0,410                 | 0,413                      |
| EasyOCR    | 11,98           | 0,379                 | 0,362                      |

**Insights**
- *Precisão*: PaddleOCR gera transcrições mais fiéis, mantendo média de ~0,78 nos caracteres.
- *Velocidade*: Tesseract é muito rápido (~0,5 s/doc) porém com perda acentuada de acurácia.
- *EasyOCR*: fica no meio-termo, com tempo próximo ao Paddle, mas menos preciso.
- *Trade-off*: utilize PaddleOCR quando qualidade é prioridade; Tesseract serve para extrações rápidas de baixa exigência.

Os arquivos exportados seguem o padrão `nomeoriginal_engine.txt` no diretório `results/`, facilitando auditorias e comparações.
