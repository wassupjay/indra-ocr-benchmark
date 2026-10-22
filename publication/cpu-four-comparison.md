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
