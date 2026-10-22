# Repeated Paddle CPU configuration check

Configuration verified on control: **True**.

Unchanged PP-OCRv6 medium weights, three original synthetic images, four CPU threads, plain Paddle backend. One first-image warmup excluded from throughput. Three measured serial passes.

| Pass | Image-to-text seconds | WER | CER |
|---|---:|---:|---:|
| 1 | 305.952 | 0.000% | 0.000% |
| 2 | 302.524 | 0.000% | 0.000% |
| 3 | 304.543 | 0.000% | 0.000% |

Median: 304.543 s per three-image pass; 0.591 control images/min. Pass-time CV: 0.57%.
Peak sampled process-tree RSS: 1197.0 MiB. Mean busy CPU cores over the whole worker: 3.43.

whole worker including imports, initialization, one warmup, all 3 measured passes, scoring and writes; summed RSS can double-count shared pages; CPU/RAM are not inference-only

three synthetic variants of one 124-word passage; configuration check, not a diverse accuracy benchmark or optimized full-book speed

Detector/recognizer weight hashes and effective runtime settings are in paddle-verification.json. Original book results were preserved. The ONNX comparison and optimal Paddle speed have not been established.
