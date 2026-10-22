Run installation from a normal PowerShell terminal in `J:\projects\indraocr`:

```powershell
& .\setup-benchmark.ps1
```

This creates `.benchmark-venv` and explicitly installs Indra 0.3.2, RapidFuzz,
the renderer/image dependencies, PaddleOCR 3.7.0 and PaddlePaddle 3.2.2 there.
It installs Tesseract 5.5.3 with winget. Internet access is required for packages
and the first PaddleOCR model download. Use the same Python environment that
already imports `indra_ocr` before running setup. No API account is required.

Run a 5-page smoke test first:

```powershell
.\.benchmark-venv\Scripts\python.exe benchmark.py --limit 5 --repeats 1
```

Then run the full comparison (114 pages in the current PDF, three timing passes):

```powershell
.\.benchmark-venv\Scripts\python.exe benchmark.py
```

Results live in `eval/benchmarks/<pdf-hash>-200dpi-all/`:

- `comparison.html`: model scores and searchable word changes.
- `comparison.json`: full results and dataset manifest.
- `medium-draft.md`: results table and methodology for an article draft.
- Per-engine logs and timestamped outputs, timing, settings and versions.

To reuse the completed Indra run and run only missing engines:

```powershell
.\.benchmark-venv\Scripts\python.exe benchmark.py --engine tesseract
.\.benchmark-venv\Scripts\python.exe benchmark.py --engine paddleocr
.\.benchmark-venv\Scripts\python.exe benchmark.py --engine summary
```

Every engine receives the same 200-DPI RGB PNGs, never the PDF. The renderer and
reference extractor run once. Indra uses `mode="ocr"` and retains furniture so
headers/footers count for all engines. Tesseract uses English, OEM 1, PSM 3.
Indra's plain `page.text` is scored, so Markdown/table markup cannot create false word errors.
PaddleOCR uses `PP-OCRv6_medium_det` and `PP-OCRv6_medium_rec` on CPU, with
orientation and unwarping disabled for these upright pages. Thread setting is 4.
Engine-specific internal resizing and preprocessing remain their defaults.
PaddleOCR uses oneDNN acceleration (`enable_mkldnn=True`) and pins its runtime
to PaddlePaddle 3.2.2. The 3.3.0 oneDNN/PIR backend crashed during warmup on this
Windows machine. The PP-OCRv6 model versions are unchanged. Turning acceleration
off was tested but did not finish a single page within a minute, so it is not used
for timing. The runtime compatibility choice is documented in the generated draft.
Upstream report: https://github.com/PaddlePaddle/Paddle/issues/77340

Models warm up once before timing. Throughput includes image decoding and OCR,
but excludes model loading, PDF rendering, scoring and writing outputs. Tesseract
CLI starts once per page: that startup is included. Python engines remain loaded.
Three measured passes report median serial throughput; first-pass output is
scored. Failures count as empty predictions and appear in results. This compares
these application integrations, not isolated model inference speed.

Both text inputs get Unicode NFC, collapsed whitespace and removal of Markdown
heading prefixes. Case, punctuation and numbers count. No lines are excluded.
Micro WER/CER sum edit counts and reference lengths; CER counts code points.

Publication limits: this is one English book with an **unverified PDF text-layer
reference**. It is a case study, not a general OCR accuracy ranking. The earlier
auto-mode Indra result is excluded because it could access the reference layer.
Check representative reference pages and use a multi-document labeled test set
before making broad accuracy claims. Do not upload the book text/images as part
of the article without the necessary rights. Nothing is published automatically.

Sources checked October 18, 2026:

- https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v3.7.0
- https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html
- https://paddlepaddle.github.io/PaddleX/latest/en/installation/paddlepaddle_install.html
- https://github.com/tesseract-ocr/tesseract/releases/tag/5.5.3

Verification: `python -m unittest test_evaluate.py test_benchmark.py`.

## Publication statistics and resource diagnostics

Generate statistics from the existing saved predictions and timings, without OCR:

```powershell
.\.benchmark-venv\Scripts\python.exe benchmark_stats.py
```

`publication-stats.md` and `.json` contain WER/CER, word edit breakdowns,
exact-text page counts, first-pass latency p50/p95, pass-time ranges,
paired engine differences and descriptive page-bootstrap intervals.
`page-stats.csv` contains each engine/page's metrics for charts.
Summary generation also updates these files automatically.

The intervals use 5,000 paired percentile resamples of page IDs, seed 20261018.
They describe within-book variation, not generalization across independent
documents. A fixed full-book score itself is not uncertain; the intervals show
variation under the page-resampling assumption. No speed confidence interval is
claimed from only three passes. Method reference:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html

Collect a bounded resource diagnostic, separately from historical full-book runs:

```powershell
.\.benchmark-venv\Scripts\python.exe profile_ocr.py --page 5 --timeout 180
```

This runs each engine sequentially, warming up on the cover then recognizing
page 5 once. It samples process-tree RSS, CPU time and thread counts every 100 ms.
It records system available RAM, actual Paddle runtime configuration and separate
detector/recognizer inference durations and input shapes. Output is timestamped
under the dataset's `profiles/` folder. The timeout applies to each engine's
whole worker. It kills only processes created by that diagnostic.

Peak RSS includes imports, model loading and the sample. Summed process RSS can
double-count shared pages; sampled CPU can miss short-lived child work. These
are **sample profiles**, not full-book peak-memory measurements. Historical CPU
and RAM use cannot be reconstructed. Do not merge sample memory and full-book
speed into one purported full-run measurement.

Additional verification: `python -m unittest test_stats.py`.

For bounded Paddle backend/thread diagnosis on the unchanged control image:

```powershell
.\.benchmark-venv\Scripts\python.exe profile_ocr.py --engines paddleocr --dataset eval/benchmarks/synthetic-control-v1 --page 3 --skip-warmup --threads 1 --timeout 240
.\.benchmark-venv\Scripts\python.exe profile_ocr.py --engines paddleocr --dataset eval/benchmarks/synthetic-control-v1 --page 3 --skip-warmup --paddle-no-mkldnn --timeout 240
```

Run them sequentially. These cold diagnostics change runtime settings, preserve
the original model/input and saved full-book results, and score the resulting
text. They do not replace warmed, repeated full-book throughput. Effective
backend and thread settings are recorded separately from the requested settings.

Run the installed plain-CPU configuration check without any downloads:

```powershell
.\.benchmark-venv\Scripts\python.exe verify_paddle.py
```

It copies the three unchanged known-text control images to a new timestamped
dataset, warms up once, and measures three passes with four requested threads.
Every pass gets numerical WER/CER scores, normalized output hashes and per-image
timings. Model hashes, actual runtime and whole-worker sampled CPU/RAM are saved.
The controller has a one-hour cap and automatically updates the article/archive.
The report is `publication/paddle-verification.md`. This verifies the tested
configuration on the small control, not maximum possible speed or full-book
throughput. The original book measurements stay separate.
