# Public annotated accuracy test

The dataset download was attempted on October 18, 2026. The agent session refused
the HTTPS connection (`WinError 10061`), so no public-dataset scores were produced.
WSL inventory was also denied. Existing book results remain unchanged.

The next dataset is [OmniDocBench](https://github.com/opendatalab/OmniDocBench),
using its dataset-provided English text annotations. The preparer selects one
ordinary text block per source page and balances selection across available
document categories. It downloads only the annotation file and selected images.
There is no manual transcription step.

Run from a normal network-enabled PowerShell terminal:

```powershell
.\.benchmark-venv\Scripts\python.exe finish_publication.py
```

This single command prepares 100 annotated regions, runs all three engines with
three passes and four CPU threads, captures whole-worker resources, audits saved
predictions, and rebuilds the article/archive. Each engine has a three-hour cap.
Successful matching runs resume; partial or failed runs are never treated as
complete. `publication/completion-status.json` records any failure.

The follow-up defaults to the tested ONNX CPU backend with denormal-as-zero
enabled, preserving converted PP-OCRv6 medium weights. `--paddle-backend plain`
retains the native plain-CPU option. This is an explicit runtime change from the
original book baseline, not a claim of optimal Paddle speed. Resume validates
actual backend/provider/threads, denormal setting and converted model hashes.
Initialization and the first warmup are excluded from throughput. Resource
measurements include the entire worker and are labeled separately.

Use `--source PATH` for an already downloaded annotation/images folder, or
`--count 20` for a smaller pilot. This is a custom region study, not a universal
accuracy ranking. The download/real-dataset integration remains unverified in
this restricted session.

Interpretation matters:

- Inputs are annotated text-region crops, **not full pages**. Throughput is text
  regions per minute. Do not compare that throughput with the book's pages/min.
- The task tests recognition with gold crop boundaries. It does not test finding
  page regions or their reading order.
- Eligibility is fixed before model inference: English pages, nonignored
  `text_block`, 20–100 words, no merged blocks or annotated inline equations,
  normal text orientation, excluding the `note` page category.
- Seed 20261018 controls sampling. One crop per selected page avoids treating all
  regions from one page as independent samples. Different pages may still come
  from the same document, and model training overlap is not known.
- This is a custom subset with CER/WER, **not an official OmniDocBench score**.
  Source annotations can contain errors; they have not been independently audited
  here. Follow the dataset's research-use terms and cite its authors.
- Source revision, annotation/image/crop hashes, crop bounds and annotation IDs
  are preserved. A dimension mismatch stops preparation rather than silently
  rescaling an annotation.

Sources:

- [Dataset and annotation schema](https://huggingface.co/datasets/opendatalab/OmniDocBench/blob/main/README.md)
- [Official evaluation code](https://github.com/opendatalab/OmniDocBench)

# Synthetic control already available locally

`eval/benchmarks/synthetic-control-v1/` contains three rendering variants of one
original 124-word passage. Exact reference text is known because that text is
drawn into each image. The renderer records font hashes, sizes and blur settings.
It validates line widths and prevents clipping. The text is original; no book
text or font files are distributed with the control.

This is a synthetic clean-print control. It cannot replace real scanned-document
ground truth. Scores from repeated font variants do not establish a general
accuracy ranking. The control has one measured pass and one first-page warmup;
resource collection covers the whole worker, including imports and loading.

Reproduction:

```powershell
.\.benchmark-venv\Scripts\python.exe control_benchmark.py
```

Each engine is bounded to eight minutes. A timeout is reported as incomplete;
it is not turned into an invented throughput or a completed accuracy score.
