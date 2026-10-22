# Paddle CPU speed investigation

## Completed full-book verification

`publication/fullbook-comparison.md` / `.json`: all three engines rerun on all
114 saved 200-DPI RGB book pages, three measured passes each and one warmup.
Paddle ONNX completed with zero failures: **2.89 pages/min**, WER **2.093%**,
CER **0.619%** against the unverified PDF text layer. Pass totals were
2369.675, 2371.192 and 2352.941 seconds (CV **0.43%**). Peak sampled whole-worker
tree RSS was **3093.4 MiB**; mean busy cores **3.59**. All three passes produced
identical normalized outputs. Actual CPU provider, four intra-op/one inter-op
threads, denormal setting, ONNX hashes and copied configuration hashes matched
the recorded conversion. No claim of maximum possible speed is made.

113/114 first-pass outputs match historical native Paddle exactly. Page 17
differs in two quote/replacement-character positions; total word/character edit
counts are unchanged. The historical native weights were not hashed, so the
difference cannot be attributed solely to the runtime conversion. Indra and
Tesseract outputs match their historical outputs on all 114 pages.

The fresh comparison measured Indra **32.38** and Tesseract **56.91 pages/min**.
Their pass CVs were **0.33%** and **1.04%**. These current measurements replace
the subset as the main deployment result; historical tables remain separate.

## Completed ONNX comparison and faster tested configuration

`publication/onnx-comparison.md` / `.json`: same source weights and three control
images, one warmup excluded, three measured passes, four CPU threads. ONNX CPU
with `session.set_denormal_as_zero=1` measured median **19.513 s** per three-image
pass, versus **304.543 s** for plain Paddle: **15.61x** observed control-set ratio.
All nine outputs match baseline exactly after normalization, zero WER/CER.
ONNX pass-time CV was 1.24%. Source hashes, actual CPU provider/threads, setting,
converted hashes and full ONNX validation were checked. No FP16 or quantization.
The default ONNX diagnostic's single measured image took 143.74 s; it was stopped
and is not presented as a complete three-pass comparison. The floating-point
setting can affect numerical results, so quality checks are necessary.

`publication/deployment-comparison.md` / `.json`: fresh three-engine comparison
on 12 fixed seeded book pages, three measured passes each. Paddle ONNX measured
**3.32 pages/min**, WER **2.489%**, CER **1.129%** against the unverified PDF text
layer. Its 12 first-pass outputs match saved historical native-Paddle outputs
exactly (5464 reference words, 136 word edits, 340 character edits). Every engine's
normalized outputs remained identical across all three passes. The new subset
table is separate from the historical 114-page table. Indra's timing CV was high
(38.77%), explicitly disclosed; Tesseract 0.55%, Paddle ONNX 0.45%.

This verifies a substantially faster tested Paddle configuration on the control
and a measured deployment on the subset. It does not establish maximum possible
speed or independent real-document accuracy. Reproduction: `ONNX_REPRO.md`.

## Completed repeated local verification

`publication/paddle-verification.md` / `.json` contain the completed control-set
check: unchanged three original images, same PP-OCRv6 medium weights, plain CPU
backend, four requested threads, one warmup excluded and three measured passes.

Pass totals: 305.952, 302.524 and 304.543 seconds. Median throughput is **0.591
control images/min**, with pass-time CV **0.57%**. Every pass had zero word and
character edits. All nine normalized output hashes match the exact references.
Actual backend/thread settings and model weight hashes were recorded.

Sampled peak process-tree RSS was 1197.0 MiB and whole-worker mean busy CPU cores
3.43. Resources include imports, loading, warmup, all passes, scoring and writes;
summed RSS can double-count shared pages. This verifies the tested configuration
on three synthetic variants of one passage. It establishes neither maximum
possible Paddle speed nor full-book throughput. The later ONNX comparison is
reported above.

Raw evidence:
`eval/benchmarks/paddle-runtime-verification/20261018T203002653377Z/`.

## Earlier single-image investigation

The saved full-book benchmark is preserved. These are separate cold diagnostics
on synthetic-control-v1 image 3, using the same PP-OCRv6 medium weights and PNG.
No resizing, quantization or model replacement was used.

| Runtime | Threads | Image-to-text seconds | Detector | Recognizer | Word edits |
|---|---:|---:|---:|---:|---:|
| oneDNN / new IR | 4 | 164.610 | 35.087 | 129.352 | 0 / 124 |
| Plain Paddle / new IR | 4 | 109.829 | 38.310 | 71.370 | 0 / 124 |
| Plain Paddle / legacy IR | 4 | 110.556 | 38.190 | 72.211 | 0 / 124 |

The plain backend reduced total time by about 33% in these individual runs.
This is a lead, not a repeated speed estimate or a proven root cause. Detection
remains slow and the outputs only validate this small known-text control.

Raw evidence under `eval/benchmarks/synthetic-control-v1/profiles/`:

- `20261018T192314149810Z/paddleocr.json`: original oneDNN control.
- `20261018T194745980620Z/paddleocr.json`: plain backend, effective runtime
  confirmed as `paddle` on both detector and recognizer, four threads, same
  image/dataset hashes recorded, completed without timeout.
- `20261018T194943275938Z/paddleocr.json`: oneDNN with one thread. Stopped
  after 135.93 seconds of whole-worker time while still in detection. This
  setting is slower; there is no completed text or total-page time to report.
- `20261018T195205585711Z/paddleocr.json`: plain backend with legacy IR.
  Completed with zero edits; no useful additional speed improvement.

Reproduction commands are in BENCHMARK.md. Historical full-book speed must not
be relabeled with this single-image result. The isolated converter and tested
ONNX runtime described above were added after these original diagnostics.
