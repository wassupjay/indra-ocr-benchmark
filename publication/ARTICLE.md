# Four-engine CPU comparison

unverified PDF text layer; scores measure agreement, not verified accuracy

Same 114 saved 200-DPI RGB page images. Four requested inference threads, one first-image warmup, three measured passes. Separate sequential engine workers on the same PC.

| Engine | WER | CER | Pages/min | Engine init (s) | Peak tree RSS (MiB) | Pass CV | Failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| Indra | 2.057% | 0.585% | 207.48 | 0.534 | 334.9 | 1.39% | 0 |
| Tesseract 5 | 2.462% | 0.679% | 58.30 | 0.042 | 117.4 | 0.39% | 0 |
| EasyOCR | 15.968% | 7.430% | 3.51 | 3.559 | 10000.9 | 0.46% | 0 |
| PaddleOCR Tiny | 3.070% | 0.763% | 77.18 | 1.477 | 1347.5 | 0.37% | 0 |

Indra is forced into OCR mode. Tesseract uses English/OEM1/PSM3 and starts one CLI process per page. EasyOCR uses English CRAFT + english_g2, CPU, greedy decoding, batch 1, zero data-loader workers and its requested default dynamic quantization; actual quantized module types are recorded. Paddle uses the official PP-OCRv6 Tiny ONNX detector/recognizer with CPUExecutionProvider, four intra-op/one inter-op threads and denormal-as-zero enabled. Tiny is not the historical medium model.

Throughput counts serial image-to-text calls, including file decode and Tesseract CLI startup. Loading, warmup, scoring and writes are excluded. Engine init is measured within each worker and includes engine imports and loading; it is not a guarantee of true cold-start latency. All engines retain scores and normalized output hashes for every pass. First-pass predictions were rescored from saved text.

CPU/RAM are sampled every 100 ms over the entire worker, including imports, loading, warmup, inference, scoring and writes. Summed RSS can double-count shared pages; sampling can miss short peaks or short-lived child CPU work. Resource and throughput scopes differ. Historical/subset timings are separate experiments. This one-book comparison does not establish a universal winner.

## Paired word-error differences

Negative differences favor the left engine. These are descriptive within-book page-bootstrap intervals (5,000 resamples, seed 20261018), not generalization estimates across independent books.

| Comparison | WER difference (pp) | Paired 95% interval (pp) |
|---|---:|---:|
| indra minus tesseract | -0.4048 | -0.4986 to -0.3180 |
| indra minus easyocr | -13.9114 | -14.5289 to -13.3063 |
| indra minus paddleocr_tiny | -1.0128 | -1.3506 to -0.6898 |
| tesseract minus easyocr | -13.5066 | -14.1267 to -12.8982 |
| tesseract minus paddleocr_tiny | -0.6080 | -0.9654 to -0.2690 |
| easyocr minus paddleocr_tiny | 12.8986 | 12.3560 to 13.4742 |

## Checks and limits

All 456 first-pass predictions were rescored from their saved text. Every measured pass has numerical quality scores and normalized output hashes; all engines produced identical normalized output across their three current passes. Runtime/model hashes and source-image/reference hashes are retained. CPU/RAM include the whole worker; engine setup is not true cold-start latency, and Tesseract loads its OCR model inside each timed CLI call.

Indra returned to roughly 207 pages/min in this run after a previous 114-page run measured 32.38 pages/min with matching model hashes. The cross-session drift remains unexplained. Keep historical measurements separate; stable pass times here do not establish universal speed. The earlier medium-Paddle comparison is preserved in fullbook-comparison.md. Tiny and medium results must not be presented as one model configuration.

Current known-text single-image probes: Indra and Tesseract had zero word/character edits; EasyOCR had 19 word and 37 character edits; Tiny had 3 word and 2 character edits. These functional probes are not a diverse ground-truth benchmark. No independently annotated real-document follow-up has completed.

MEDIUM_POST.md is the reader-facing four-engine draft. The commercial PDF, source images, reference text and raw book predictions stay out of the release archive. Nothing has been posted to Medium.

## Supplied transcription follow-up

[Candidate-reference comparison](transcription-comparison.md) includes all 114 pages, a 99-page sensitivity analysis excluding all flagged pages, and quote-style normalization checks. The supplied transcription is not independently verified ground truth; original PDF-reference scores above are preserved. Speed and RAM were not remeasured.
