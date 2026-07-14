# Models & Finetuning

This document explains the digit-recognition models used by the sudoku
solver, the path we took to get to the final finetuning recipe, the
orientation/language auto-detection logic used at inference time, and the
ONNX export used to make the API fast enough to serve.

> All experimentation notebooks (the actual training runs, plots, dataset
> sanity checks, etc.) live in **`/notebooks`**, which is **gitignored** and
> not part of this repo's history. This document is the write-up of what was
> done there; the notebooks themselves are local-only scratch work.

## 1. Model architecture

Both the English and Persian digit classifiers share the exact same
architecture, defined once in `src/recognition/model.py`:

- Backbone: `torchvision.models.efficientnet_b1`, ImageNet-pretrained
  (`EfficientNet_B1_Weights.DEFAULT`).
- Classifier head replaced with:
  `Dropout(0.3) -> Linear(1280, 512) -> Dropout(0.2) -> Linear(512, 10)`
  (10 classes: digits 0–9, where `0` means "no digit / empty cell").
- Input: 224x224 RGB, normalized with `mean=0.5, std=0.5`.

The only difference between the English and Persian models is which weights
they're finetuned on — same code, two checkpoints
(`src/models/best_model_english.pth`, `src/models/best_model_persian.pth`).

## 2. English digit model: three attempts

We iterated through three datasets before the English digit model was good
enough. All three used the identical architecture and the same 2-stage
finetuning recipe (§4) — only the **training data** changed between
attempts.

### Attempt 1 — Chars74K "Fnt" (typed characters)

`data/English/Fnt/` is the Chars74K synthetic **typed-font** character
dataset. We started here because Sudoku digits are printed/typed, not
handwritten, so typed-font characters seemed like the closest match to what
we'd actually see in a Sudoku grid photo.

**Result: not good enough.** After the grid-extraction preprocessing
(`src/grid_extraction.py` — thresholding, cropping to the digit's
connected component, padding to a square, resizing to 224x224, see §6), our
extracted digit crops looked meaningfully different from clean, evenly-lit,
perfectly-cropped typed-font glyphs. There's noise, uneven lighting,
imperfect binarization, and cropping artifacts around the digit. The result
was a **train/serve skew**: the model was strong on clean typed digits but
weak on our actual, preprocessed cell crops.

### Attempt 2 — MNIST (handwritten digits)

Next we tried `data/mnist/` (standard MNIST handwritten digits). This did
better in practice than the Chars74K attempt, but it still didn't satisfy
us. Same root cause as before: **train/serve skew**. MNIST digits are
centered, denoised, and normalized in a way our real preprocessed cell crops
aren't — different stroke statistics, different framing, different noise
profile.

### Attempt 3 (final, production) — finetune directly on preprocessed Sudoku cells

The fix was to stop trying to find a digit dataset that merely *looks like*
Sudoku digits, and instead build a dataset that **is** the output of our own
preprocessing pipeline:

1. Take a Sudoku puzzle image dataset (Kaggle
   [`mexwell/sudoku-image-dataset`](https://www.kaggle.com/datasets/mexwell/sudoku-image-dataset/data),
   stored under `data/sudoku/` — `v1_training`, `v2_train`, etc., with
   corner annotations in `outlines_sorted.csv`).
2. Run each image through `src/grid_extraction.py` (the exact same code
   path used in production) to warp, split into 81 cells, and clean each
   digit crop.
3. Pair each extracted cell with its ground-truth label to build a
   per-class folder dataset: `data/digit_finetune_dataset/{train,test}/<digit 0-9>/*.png`.
4. Finetune EfficientNet-B1 on *that* dataset.

This closes the train/serve gap completely, because the training images are
produced by the identical pipeline the model sees at inference time. This is
the approach that "hit the nail" — it's what `src/models/best_model_english.pth`
is trained on and what the API actually serves.

## 3. Persian digit model

Same architecture, same 2-stage recipe, but a different data story since
there's no ready-made "Persian Sudoku" dataset:

1. **Handwritten digits — Hoda dataset.** `data/hoda-dataset/DigitDB/`
   (`Train 60000.cdb` / `Test 20000.cdb`, read via
   `notebooks/HodaDatasetReader.py`) is a standard Persian handwritten-digit
   dataset. Finetuning on it gave results comparable to the English MNIST
   attempt — decent, but not good enough on our real preprocessed crops, for
   the same train/serve-skew reason as English attempt #2.
2. **No Persian Sudoku dataset exists**, so we built one synthetically,
   using the scripts in `/scripts`:
   - `scripts/persian_digit_dataset.py` generates fake "photographed"
     Persian Sudoku puzzles from scratch: it lays out a random 9x9 grid,
     renders Persian digits (۱–۹) with a real Persian font
     (`scripts/Yekan.ttf`), warps the clean puzzle onto a randomized
     perspective (simulating a photo taken at an angle), and then applies
     randomized degradations — subtle rotation/scale, low light, Gaussian
     noise, blur, and directional shadow gradients — to mimic real photo
     conditions.
   - `scripts/build_persian_finetune_dataset.py` then runs those synthetic
     images through the **same** `src/grid_extraction.py` pipeline used in
     production (exactly like step 2/3 of the English attempt #3 above),
     producing a labeled per-class dataset at
     `data/persian_digit_finetune_dataset/{train,test}/<digit 0-9>/*.png`.
3. Finetuning on this synthetic-but-pipeline-matched dataset is what
   actually worked well for Persian, mirroring what worked for English:
   match the training distribution to the *preprocessed* distribution, not
   to a generic "clean digits" dataset.

(`best_model_persian_soft.pth` / `best_model_persian_hard.pth` in
`src/models/` are earlier checkpoints from softer/harder augmentation
variants of this dataset, kept around for comparison; `best_model_persian.pth`
is the one actually used.)

## 4. Finetuning convention

For all digit-recognition finetuning runs (`src/recognition/train.py`) we
followed the transfer-learning convention described in the TensorFlow guide
[Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning),
adapted to PyTorch/EfficientNet:

- **BatchNorm layers are never trained.** Per the guide's recommendation,
  we do *not* let BatchNorm layers update their running statistics during
  finetuning — updating BN stats on a small finetuning set with a different
  distribution than the original pretraining set tends to hurt more than it
  helps. In `train.py`, every `BatchNorm2d` module is forced into `.eval()`
  mode inside the training step (even while the rest of the model is in
  `.train()` mode), so BN layers always run in inference mode and are
  excluded from training regardless of which parameters are otherwise
  unfrozen.
- **Two-stage finetuning:**
  1. **Stage 1 — frozen backbone.** `model.features` (the EfficientNet-B1
     backbone, starting from ImageNet weights) is fully frozen
     (`requires_grad = False`). Only the new classifier head is trained,
     with `Adam(lr=1e-3)` for 6 epochs. This lets the new head adapt to the
     10-class digit output without disturbing the pretrained features.
  2. **Stage 2 — partial backbone finetune.** The last backbone block
     (`model.features[-3]`) is unfrozen alongside the classifier head, and
     training continues with a much lower learning rate
     (`Adam(lr=1e-5)`) for 10 epochs, keeping the best checkpoint by
     validation accuracy. BatchNorm layers remain frozen throughout (see
     above).

This mirrors the guide's "train head first, then unfreeze part of the
backbone at a low LR" pattern, with the BN-frozen caveat applied throughout
both stages.

## 5. Inference-time auto-detection: orientation & language

The Sudoku photo the API receives could be rotated 0/90/180/270 degrees and
could be either an English or a Persian puzzle. Both are resolved the same
way — by brute-force voting with the recognition model itself, in
`src/orientation/infer.py::resolve_orientation` and
`src/api/app.py::solve_sudoku`:

**Orientation (0/90/180/270):** for each candidate rotation, the grid is
re-cropped into 81 cells (`rotate_extraction` in `src/grid_extraction.py`,
which re-derives the cell split from the warp geometry rather than just
rotating pixels), every non-empty cell is classified, and we take the
**average confidence across all non-empty predictions**. The rotation with
the highest mean confidence wins. This is done independently for the
English and the Persian model.

**Language (English vs. Persian):** with the best rotation found for each
language, we score each language's full-grid prediction with
`_grid_score()` in `src/api/app.py` and pick whichever language's grid has more
confidently-recognized digits (ties broken by confidence). Whichever
language wins also determines the orientation used, since orientation was
resolved per-language.

### Why not a dedicated orientation classifier?

We also tried training a dedicated 4-class rotation classifier
(`src/orientation/model.py` — a ResNet18 head predicting `{0°, 90°, 180°,
270°}`, trained via `src/orientation/train.py`) instead of the brute-force
voting approach above. **It didn't work well enough to use.**

The core problem: several digits are near-mirror-images of their own 180°
rotation — most notably **6/9 in English** and **7/8 in Persian** (۷/۸) —
so a model looking at the whole warped grid can't always tell orientation
apart from digit identity confusion, and a purely visual/geometric
orientation classifier struggles on exactly the cases where it matters
most. The brute-force "try all 4 rotations, keep whichever gives the
recognition model the highest confidence" approach sidesteps this because
it lets the digit-recognition model itself (which is what we actually care
about getting right) be the tie-breaker, rather than trusting a separate,
weaker orientation signal. The orientation classifier code is kept in the
repo but is not wired into the production `/solve` endpoint.

## 6. Grid extraction / preprocessing (context for §2–3)

Not a model, but relevant since it's the thing that both defines "what the
finetuning data must look like" and runs at inference time:
`src/grid_extraction.py` locates the Sudoku grid in a photo (adaptive
thresholding + contour/Hough-line detection), perspective-warps it to a
fixed 450x450 canvas, removes the grid lines, locates each of the 81 cells
(with a curved-line-tracing fallback for skewed/folded photos), isolates
the digit's connected component per cell (filtering out noise/smudges by
size, aspect ratio, and fill density), and normalizes/pads/resizes it to
224x224. This is the exact function reused by `build_persian_finetune_dataset.py`
and the equivalent English dataset-building step to make sure training data
matches serving data pixel-for-pixel.

## 7. ONNX export

### What ONNX is and why we use it

ONNX (Open Neural Network Exchange) is a portable model format: instead of
serving a PyTorch model that needs the full PyTorch runtime (Python,
autograd machinery, dynamic graph tracing, etc.) to run a forward pass, the
trained model's computation graph is exported once to a static `.onnx`
file and executed by **ONNX Runtime** (`onnxruntime`), a lightweight,
graph-optimized inference engine (operator fusion, constant folding, better
CPU kernel selection) with no PyTorch dependency at request time.

`src/recognition/onnx_exporter.py` does the export:

- Loads the finetuned `.pth` weights into the EfficientNet-B1 architecture.
- Traces it with `torch.jit.trace` and saves a TorchScript `.pt` file
  (an intermediate comparison point).
- Exports it to ONNX via `torch.onnx.export` (opset 17, dynamic batch
  axis), producing `src/models/best_model_english.onnx` and
  `src/models/best_model_persian.onnx`.
- Benchmarks all three formats (PyTorch `.pth`, TorchScript `.pt`, ONNX
  Runtime) on 100 forward passes each (10 warmup runs) and prints a
  size/latency comparison table.

`src/recognition/onnx_infer.py` is the ONNX-Runtime counterpart of
`src/recognition/infer.py`: it loads the two `.onnx` files into
`onnxruntime.InferenceSession` objects (CPU execution provider) and exposes
the same `predict_cell` / `predict_cell_proba` interface, so the rest of
the codebase doesn't need to know which backend is active.

### Result (`onnx_result.png`)

![ONNX benchmark result](./onnx_result.png)

Running `python -m src.recognition.onnx_exporter` re-exports both language
models and prints this comparison:

| Format             | English size | English latency | Persian size | Persian latency |
|---------------------|-------------:|-----------------:|--------------:|------------------:|
| PyTorch (`.pth`)    | 27.78 MB     | 411.32 ms         | 27.79 MB       | 411.84 ms          |
| TorchScript (`.pt`) | 28.41 MB     | 413.97 ms         | 28.41 MB       | 400.17 ms          |
| **ONNX Runtime**    | **27.35 MB** | **12.99 ms**      | **27.35 MB**   | **11.82 ms**       |

TorchScript tracing alone bought us essentially nothing over eager PyTorch
(~400ms either way) — the bottleneck isn't graph tracing, it's the PyTorch
CPU execution path itself. ONNX Runtime, on the same weights and the same
CPU, is **~30–35x faster** (≈411ms → ≈12ms per single-image forward pass),
with a slightly *smaller* file than either PyTorch format.

**Why this matters for this project specifically:** a single `/solve`
request classifies up to **81 cells**. At ~410ms/cell (PyTorch), a full
grid would take roughly **81 x 0.41s ≈ 33 seconds** per request — not
viable for a synchronous HTTP endpoint, especially since orientation
resolution alone runs the recognition model over up to 4 candidate
rotations per language before a grid is even finalized. At ~12ms/cell
(ONNX), the same 81 cells take **≈1 second**. This benchmark is the reason
`src/api/app.py` sets `RECOGNITION_BACKEND = "onnx"` and loads
`src/recognition/onnx_infer.load_model()` at startup instead of the raw
`.pth` weights — the ONNX models are what the deployed API actually runs.

## 8. File/folder reference

```
src/
  models/                        # gitignored (*.pth, *.pt, *.onnx) — trained weights & exports
    best_model_chars.pth             # attempt 1: EfficientNet-B1 finetuned on Chars74K typed chars — superseded
    best_model_mnist.pth             # attempt 2: EfficientNet-B1 finetuned on MNIST handwritten digits — superseded
    best_model_english.pth           # attempt 3 (final): finetuned on preprocessed Sudoku-cell crops — production English weights
    best_model_persian.pth           # final Persian weights: Hoda + synthetic Persian-Sudoku crops
    best_model_persian_soft.pth      # Persian checkpoint, softer augmentation variant — kept for comparison
    best_model_persian_hard.pth      # Persian checkpoint, harder augmentation variant — kept for comparison
    best_model_english.pt            # TorchScript export of best_model_english.pth
    best_model_persian.pt            # TorchScript export of best_model_persian.pth
    best_model_english.onnx          # ONNX export of best_model_english.pth — loaded by the API at runtime
    best_model_persian.onnx          # ONNX export of best_model_persian.pth — loaded by the API at runtime

  recognition/                   # digit classifier: architecture, training, inference (torch + onnx)
    model.py                         # EfficientNet-B1 + custom classifier head; loads .pth weights
    dataset.py                       # DigitFinetuneDataset — reads train|test/<0-9>/*.png folder-per-class layout
    train.py                         # 2-stage finetuning loop (frozen backbone -> partial unfreeze), BN kept frozen
    infer.py                         # predict_cell / predict_cell_proba — dispatches to torch or ONNX backend
    onnx_exporter.py                 # exports .pth -> TorchScript (.pt) + ONNX (.onnx); benchmarks all 3 formats
    onnx_infer.py                    # ONNX Runtime session wrapper, same interface as infer.py

  orientation/                   # 0/90/180/270 rotation classifier (experimental — NOT used in production, see §5)
    model.py                         # ResNet18, 4-class head
    dataset.py                       # RotationDataset — rotates upright crops on the fly to synthesize labels
    train.py                         # 2-stage finetune (frozen backbone -> full unfreeze)
    infer.py                         # OrientationCorrector / WarpOrientationClassifier + resolve_orientation()
                                      #   (resolve_orientation is the brute-force voting fn actually used by the API)

  grid_extraction.py             # locates grid, perspective-warps, splits into 81 cells, cleans/normalizes each digit
  pipeline.py                    # end-to-end orchestration (currently a stub)
  overlay.py                     # renders the solved digits back onto the original photo
  solver.py                      # Sudoku backtracking solver
  api/app.py                     # FastAPI service — wires grid_extraction -> orientation/language resolution ->
                                  #   recognition (ONNX) -> solver -> overlay behind the /solve endpoint

scripts/                        # one-off dataset-building & QA scripts — not imported by the served app
  persian_digit_dataset.py         # synthesizes fake "photographed" Persian Sudoku images (Yekan.ttf font +
                                    #   perspective warp + noise/blur/shadow) — raw material for Persian dataset
  build_persian_finetune_dataset.py# runs those images through grid_extraction.py to build a labeled per-cell
                                    #   train/test dataset — the "preprocess stage" step described in §3
  robustness_demo.py               # synthesizes an English puzzle and stress-tests grid_extraction.py under
                                    #   blur / noise / rotation / low light / shadow
  Yekan.ttf                        # Persian font asset used by persian_digit_dataset.py

data/                            # gitignored — raw & derived datasets
  English/Fnt/                     # Chars74K typed-character dataset (attempt 1)
  mnist/                           # MNIST handwritten digits (attempt 2)
  hoda-dataset/DigitDB/            # Hoda Persian handwritten-digit dataset (.cdb, read by notebooks/HodaDatasetReader.py)
  sudoku/                          # raw Kaggle "sudoku-image-dataset" photos + corner annotations (outlines_sorted.csv)
  digit_finetune_dataset/          # final English per-cell dataset (train/test/<0-9>/*.png), built via grid_extraction
  persian_digit_finetune_dataset/  # final Persian per-cell dataset, built from the synthetic Persian Sudoku images

notebooks/                       # gitignored — ALL experimentation and test notebooks live here
  test_digits.ipynb                # attempt 1 experiments (Chars74K)
  test_mnist.ipynb, "test_mnist copy.ipynb"  # attempt 2 experiments (MNIST)
  test_digits_persian.ipynb        # Persian finetuning experiments
  test_hoda.ipynb                  # Hoda dataset exploration
  test.ipynb                       # misc/scratch
  HodaDatasetReader.py             # parses the Hoda .cdb binary dataset format

docs/
  onnx_result.png                  # benchmark screenshot referenced in §7
  MODELS_AND_FINETUNING.md         # this document
```
