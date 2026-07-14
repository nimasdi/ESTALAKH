# Estalakh — Sudoku Solver from a Photo

Point a camera at a Sudoku puzzle — printed, English or Persian digits, at
any rotation, on a flat or curved page — and get back the solved grid
overlaid on the original photo.

## How it works

```
photo -> grid detection & perspective warp -> 81 cell crops
      -> orientation + language auto-detection (brute-force voted by the recognition model itself)
      -> digit recognition (EfficientNet-B1, ONNX Runtime)
      -> backtracking solver -> solution overlaid back onto the photo
```

- **Grid extraction** (`src/grid_extraction.py`) locates the grid via a
  three-tier fallback (contour → relaxed contour → Hough lines), perspective-warps
  it to a fixed canvas, and — for curved/photographed book pages — traces
  each of the 20 grid lines as a curve and cuts all 81 cells with individual
  homographies instead of one uniform grid.
- **Digit recognition** (`src/recognition/`) is EfficientNet-B1, finetuned
  separately for English and Persian digits directly on the output of the
  extraction pipeline (not a generic digit dataset), which is what closed
  the train/serve gap.
- **Orientation & language** are resolved together: all 4 rotations are
  tried for both languages, and whichever combination the recognition model
  reads with the highest average confidence wins.
- **ONNX Runtime** serves the recognition models (~12ms/cell vs. ~410ms/cell
  in eager PyTorch), which is what makes classifying up to 81 cells per
  request viable inside a synchronous HTTP call.

See [`docs/project-document.md`](docs/project-document.md) and
[`docs/MODELS_AND_FINETUNING.md`](docs/MODELS_AND_FINETUNING.md) (Persian)
for the full write-up, and [`src/GRID_EXTRACTION.md`](src/GRID_EXTRACTION.md)
for a step-by-step walkthrough of the extraction code.

## Getting started

Requires Python 3.12+.

```bash
uv sync            # or: pip install -e .
uv run main.py      # or: uvicorn src.api.app:app --reload
```

The API serves a minimal web UI at `http://localhost:8000/`.

## API

| Endpoint            | Method | Description                                                  |
| -------------------- | ------ | -------------------------------------------------------------- |
| `/`                   | GET    | Web UI                                                          |
| `/health`             | GET    | Liveness check                                                  |
| `/predict/cell`       | POST   | Classify a single cropped digit image                          |
| `/solve`              | POST   | Full pipeline: upload a puzzle photo, get grid + solution + overlay image |

`/solve` accepts `?debug=true` (dumps intermediate crops to `out/solve/`)
and `?stages=true` (returns base64-encoded intermediate pipeline images).

## Project layout

```
src/
  grid_extraction.py     # grid detection, perspective warp, cell extraction
  solver.py               # Sudoku backtracking solver
  overlay.py               # renders the solution back onto the photo
  recognition/              # digit classifier: model, training, torch/ONNX inference
  orientation/               # experimental dedicated rotation classifier (not used in production)
  api/app.py                  # FastAPI service tying the pipeline together

scripts/                # dataset-building scripts (synthetic Persian Sudoku generation, etc.)
ui/                      # minimal web frontend served by the API
docs/                    # write-ups and diagrams referenced above
```

`data/` and `src/models/` (trained weights) are gitignored; see
[`docs/MODELS_AND_FINETUNING.md`](docs/MODELS_AND_FINETUNING.md) for how
they're produced.
