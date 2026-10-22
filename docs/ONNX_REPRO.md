# Reproduce the tested Windows CPU deployment

Python 3.11, Windows x64. Start with the pinned benchmark requirements and the
original synthetic controls included in the review archive. The commercial book
is not distributed; its comparison requires the matching legally obtained PDF
and prepared page images. Source PDF and dataset hashes identify the input.

Download these converter packages in a network-enabled terminal:

```powershell
.\.benchmark-venv\Scripts\python.exe -m pip download --dest .package-cache paddle2onnx==2.1.0 onnx==1.17.0
.\.benchmark-venv\Scripts\python.exe -m pip download --no-deps --index-url https://www.paddlepaddle.org.cn/packages/nightly/cpu/ --dest .package-cache paddlepaddle==3.0.0.dev20250426
```

The exact converter runtime matters: stable 3.2.2 and nightly 3.4 failed native
symbol resolution in this installation. The converter release's build dependency
is pinned in its [official pyproject.toml](https://github.com/PaddlePaddle/Paddle2ONNX/blob/v2.1.0/pyproject.toml).

```powershell
.\.benchmark-venv\Scripts\python.exe convert_onnx.py
.\.benchmark-venv\Scripts\python.exe compare_onnx.py --flush-denormals
.\.benchmark-venv\Scripts\python.exe run_deployment_comparison.py
.\.benchmark-venv\Scripts\python.exe run_deployment_comparison.py --all-pages
.\.benchmark-venv\Scripts\python.exe publication_report.py
.\.benchmark-venv\Scripts\python.exe export_results.py
```

Conversion checks wheel hashes, unpacks the packages into isolated project
folders, loads the matching Paddle DLL and uses Paddle2ONNX's native export API.
It does not import the old Paddle Python wrapper. This is the API used by
`paddle2onnx.convert.export` for already-JSON models with decomposition disabled.
Both detector and recognizer pass full ONNX checking. No quantization or FP16.
Source/model/config/wheel hashes and converter flags are saved in
`.onnx-models/PP-OCRv6_medium/conversion.json`. The measured runtime retains the
original PaddleOCR/PaddleX preprocessing and postprocessing, and uses CPU ONNX
sessions with four intra-op threads and one inter-op thread.

`session.set_denormal_as_zero=1` is a documented ONNX Runtime CPU option. It can
change numerical outputs. The comparison scores every pass and compares normalized
text hashes with the native control. The 12-page follow-up uses a fixed random
seed and does not select pages by error score. These are tested configurations,
not proof of maximum performance or general accuracy. Current/historical tables
are different runs and sample sizes; preserve their scopes.

`--all-pages` reruns all three engines on all 114 prepared book pages, with one
warmup and three measured passes each. It creates a separate timestamped
`fullbook-onnx-114` dataset and `publication/fullbook-comparison.json`; it does
not replace the earlier native or 12-page experiments. Whole-worker CPU/RAM
sampling accompanies each engine. Allow up to three hours per engine.

The separate annotated-region continuation defaults to this tested ONNX
configuration. `finish_publication.py --paddle-backend plain` retains the plain
runtime option. Resume checks include backend, CPU provider, thread settings,
denormal setting and converted weight hashes; a different configuration reruns.

The benchmark environment snapshot records the actual imported OpenCV version
as well as installed distributions. Both OpenCV distributions are present in this
installation; package versions alone do not resolve which cv2 files were imported.

Observed Indra model snapshot: `datalab-to/indra-ocr` revision
`ed345c7f158eddbc7dee0c6a7269afb75a94f7e8`. Compare the model SHA256 values
in the measured result metadata when reproducing; a package version alone does
not establish model-weight identity. This release records the observed mixed
OpenCV installation; a clean installation has not been separately benchmarked.
