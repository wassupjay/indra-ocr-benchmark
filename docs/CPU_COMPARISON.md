# Indra vs Tesseract vs EasyOCR vs PP-OCRv6 Tiny

Run this one command from `J:\projects\indraocr` in a terminal with normal
download access:

```powershell
.\.benchmark-venv\Scripts\python.exe run_cpu_comparison.py --setup
```

It installs EasyOCR 1.7.2 with the official PyTorch 2.8.0 CPU / torchvision
0.23.0 CPU pairing into the project-local `.easyocr-venv`, reusing the pinned
benchmark dependencies through a `.pth` file. The existing benchmark environment
is not upgraded. It downloads EasyOCR's CRAFT and English-generation-2 weights
and Paddle's official Tiny ONNX packages. Torch is a pinned inference dependency;
this does not claim it is the newest Torch release.

The Tiny downloader resolves each official Hugging Face model to an immutable
revision, validates its model name, and records model/config SHA256 hashes in
`.ocr-models/PP-OCRv6_tiny/provenance.json`. No Tiny conversion is performed.
EasyOCR checks its packaged model checksums when loading; the benchmark records
SHA256 hashes and disables model downloads in measured workers.

After setup, the command probes every engine on a known-text synthetic image,
then runs all four sequentially on the same 114 saved 200-DPI RGB book images.
Each engine gets four requested inference threads, one first-page warmup and
three measured passes. EasyOCR requests its default CPU dynamic quantization;
actual quantized module types are recorded. Tiny uses CPU ONNX Runtime with
four intra-op/one inter-op threads and denormal-as-zero enabled. It is distinct
from the previously tested PP-OCRv6 medium model.

The entire worker for each engine is capped at three hours and sampled for
process-tree CPU/RAM every 100 ms. Only the controller's own worker tree is
terminated on timeout. Logs and predictions are retained under
`eval/benchmarks/cpu-four-114/`. New results are never mixed into historical runs.
Matching successful runs resume only if dataset/runner hashes, model/runtime
metadata, thread count, three pass totals and successful resource logs match.

The comparison is promoted only after all four workers complete without
execution failures and their saved predictions are rescored. Output files:

- `publication/cpu-four-comparison.md` and `.json`: WER/CER, pages/min, engine
  initialization, RAM, pass variation, full CPU/resource data and evidence audit.
- `publication/cpu-four-results.png`: shareable results table.
- `publication/cpu-four-page-stats.csv`: numerical per-page results.
- `publication/cpu-four-status.json`: actual completion state and any error.

Throughput excludes imports/loading, warmup, rendering, scoring and writes.
Tesseract's per-image CLI startup is included; Python models stay loaded.
Initialization is an in-process import/load measurement, not a guaranteed cold
start. CPU/RAM cover the whole worker and have a different scope from throughput;
summed RSS can double-count shared memory and sampling can miss short peaks.

The reference remains unverified embedded PDF text. These are agreement scores
on one book, not independently verified general OCR accuracy. Source book images,
text and predictions stay local. The Medium draft now includes the completed
four-engine study; historical medium-model experiments remain separate.

## Completed October 19, 2026

The user ran setup in a normal terminal after this agent's download attempt was
refused. All four engines completed all 114 pages and three measured passes with
zero execution failures. Dataset:
`eval/benchmarks/cpu-four-114/20261019T151429710767Z`. All four produced identical
normalized outputs across their measured passes. Indra measured 207.48 pages/min,
Tesseract 58.30, EasyOCR 3.51 and Tiny 77.18. Full metrics, configurations and
resource measurements are in `publication/cpu-four-comparison.json`.

The saved first-pass predictions were independently rescored after the user run;
runtime metadata matched the preflight probes, and actual Tiny/EasyOCR model
hashes matched their recorded files. The earlier stale setup-error field was
removed from the completed status. Existing completed runs need no rerun.

`--check` verifies local model files and dependency imports without changing a
completed run's state. `--setup` is for a fresh setup or missing dependencies;
ordinary continuation can use `run_cpu_comparison.py` and resume matching runs.

Sources:

- [EasyOCR CPU use and installation](https://github.com/JaidedAI/EasyOCR)
- [Official PyTorch CPU version pairings](https://pytorch.org/get-started/previous-versions/)
- [Official Tiny detector ONNX package](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_det_onnx)
- [Official Tiny recognizer ONNX package](https://huggingface.co/PaddlePaddle/PP-OCRv6_tiny_rec_onnx)
