# Paddle versus ONNX Runtime: CPU control comparison

Same source weights, image hashes, preprocessing, one excluded warmup and three measured passes. Four requested inference threads; ONNX actual CPU provider/thread settings verified.

ONNX session.set_denormal_as_zero: 1. This floating-point setting can affect numerical outputs; quality and text-hash comparisons below are required.

| Runtime | Median three-image pass (s) | Control images/min |
|---|---:|---:|
| Plain Paddle | 304.543 | 0.591 |
| ONNX Runtime | 19.513 | 9.225 |

Observed time ratio on this control: 15.61x. ONNX pass-time CV: 1.24%.
Zero WER/CER on every ONNX pass: True. All normalized outputs match baseline: True.
Sampled ONNX peak tree RSS: 1357.6 MiB. Whole-worker mean busy cores: 3.60.

three synthetic variants of one original passage; separate sequential runs, not interleaved; a control-set runtime comparison, not universal accuracy/speed

whole worker including imports, loading, warmup, all three passes, scoring and writes; summed RSS may double-count shared memory

Conversion used the same Paddle2ONNX native export API called by its Python wrapper, with the exact matching isolated Paddle build. FP16/quantization disabled. Full ONNX model checks passed; source and converted model hashes are recorded.
