# Fresh 12-page deployment comparison

12 sampled pages of one English book; PDF text-layer agreement, not independently verified accuracy; separate sequential engine runs, not interleaved

Three measured passes per engine after one first-page warmup. All engines receive the same saved 200-DPI RGB images. Four requested inference threads. Tesseract uses one CLI process per image; Python models remain loaded.

| Engine | WER | CER | Pages/min | Median 12-page pass (s) | Peak tree RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| indra | 2.489% | 0.986% | 35.27 | 20.416 | 296.2 |
| tesseract | 2.635% | 1.239% | 67.57 | 10.656 | 112.8 |
| paddleocr | 2.489% | 1.129% | 3.32 | 216.732 | 3037.1 |

PP-OCRv6 medium, converted same cached weights, ONNXRuntime CPU, four intra-op/one inter-op threads, session.set_denormal_as_zero=1

PDF pages: [9, 15, 26, 43, 59, 65, 70, 100, 101, 102, 110, 111]. Selection seed: 20261018.

RAM/CPU cover the entire worker, including initialization, warmup and scoring. Throughput excludes those stages. They are different measurement scopes. Summed process-tree RSS can double-count shared pages.

Raw predictions were rescored and all input/output hashes checked. Numerical scores and normalized text hashes were saved for every pass. Original 114-page results remain separate.



### Timing repeatability and resource scope

| Engine | Pass range (s) | Pass-time CV | Peak sampled tree RSS (MiB) |
|---|---:|---:|---:|
| indra | 9.377–21.083 | 38.77% | 296.2 |
| tesseract | 10.581–10.698 | 0.55% | 112.8 |
| paddleocr | 215.709–217.644 | 0.45% | 3037.1 |

Indra's first pass was materially faster than its later passes in this run. That timing variation remains unexplained; the median is reported with the range and CV, and should not be presented as a stable universal speed estimate. CPU/RAM cover the whole worker, including warmup and scoring. Throughput counts image-to-text calls only.
