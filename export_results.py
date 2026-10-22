"""Bundle code, aggregate metrics and original synthetic controls; no book text."""

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import zipfile

import cv2

from evaluate import sha256


def main():
    root = Path(__file__).resolve().parent
    book = Path("eval/benchmarks/f960d85713a3-200dpi-all")
    control = Path("eval/benchmarks/synthetic-control-v1")
    files = [Path(name) for name in (
        "benchmark.py", "evaluate.py", "benchmark_stats.py", "profile_ocr.py",
        "control_benchmark.py", "validate_control.py", "prepare_public.py", "export_results.py",
        "benchmark-requirements.txt", "setup-benchmark.ps1", "docs/BENCHMARK.md",
        "docs/PUBLIC_BENCHMARK.md", "tests/test_evaluate.py",
        "tests/test_benchmark.py", "tests/test_stats.py", "tests/test_controls.py",
        "publication_report.py", "finish_publication.py", "tests/test_publication.py", "docs/PADDLE_SPEED.md",
        "verify_paddle.py", "tests/test_verify_paddle.py",
        "convert_onnx.py", "compare_onnx.py", "run_deployment_comparison.py", "tests/test_onnx.py",
        "docs/ONNX_REPRO.md", "docs/EVALUATION.md",
        "run_cpu_comparison.py", "tests/test_cpu_comparison.py", "docs/CPU_COMPARISON.md",
        "evaluate_transcription.py", "tests/test_transcription.py",
        "publication_charts.py", "README.md", ".gitignore",
        "publish-github.ps1", "tests/test_publish_github.ps1",
    )]
    files += [book / name for name in ("dataset.json", "publication-stats.md", "publication-stats.json", "page-stats.csv")]
    files.append(Path("eval/config.json"))
    files += [control / name for name in ("dataset.json", "source.txt", "quality-validation.md", "quality-validation.json")]
    files += [control / f"page_{n:03}.{extension}" for n in range(1, 4) for extension in ("png", "txt")]
    for name in ("ARTICLE.md", "index.html", "README.md", "book-results.png", "book-results.csv", "evidence-audit.json", "completion-status.json", "paddle-verification.json", "paddle-verification.md",
                 "onnx-comparison.json", "onnx-comparison.md", "onnx-comparison-status.json", "onnx-book-quality-check.json", "deployment-comparison.json", "deployment-comparison.md", "deployment-status.json", "deployment-results.png", "deployment-page-stats.csv",
                 "fullbook-comparison.json", "fullbook-comparison.md", "fullbook-status.json", "fullbook-results.png", "fullbook-page-stats.csv", "fullbook-quality-check.json",
                 "MEDIUM_POST.md", "MEDIUM_PUBLISHING_KIT.md", "medium-post.html", "medium-cover.png", "NOTION_UPDATE.md",
                 "MEDIUM_FULL_METHODS.md", "chart-speed.png", "chart-word-errors.png", "chart-memory.png", "chart-data.json",
                 "cpu-four-status.json", "cpu-four-probe-indra.json", "cpu-four-probe-tesseract.json",
                 "cpu-four-comparison.json", "cpu-four-comparison.md", "cpu-four-results.png", "cpu-four-page-stats.csv",
                 "transcription-comparison.json", "transcription-comparison.md", "transcription-page-stats.csv"):
        candidate = Path("publication") / name
        if candidate.is_file():
            files.append(candidate)
    quality = json.loads((control / "quality-validation.json").read_text(encoding="utf-8"))
    for model in quality["models"].values():
        for item in model["items"]:
            prediction = Path(item["prediction_file"])
            if not prediction.resolve().is_relative_to((root / control).resolve()):
                raise ValueError("Control prediction export must stay inside the original synthetic corpus")
            files.append(prediction)
    for tag in ("20261018T192314149810Z", "20261018T194745980620Z", "20261018T194943275938Z", "20261018T195205585711Z"):
        files.append(control / "profiles" / tag / "paddleocr.json")
    verification_file = root / "publication/paddle-verification.json"
    if verification_file.is_file():
        verification = json.loads(verification_file.read_text(encoding="utf-8"))
        verification_folder = Path(verification["dataset"])
        if not verification_folder.resolve().is_relative_to((root / "eval/benchmarks/paddle-runtime-verification").resolve()):
            raise ValueError("Verification export must stay inside its synthetic control directory")
        result_file = Path(verification["results_file"])
        if not result_file.resolve().is_relative_to(verification_folder.resolve()):
            raise ValueError("Verification result is outside its dataset")
        files += [verification_folder / name for name in ("dataset.json", "resources.json", "verification.json", "verification-status.json")]
        files += [result_file] + list(result_file.parent.glob("page_*.txt"))
    onnx_file = root / "publication/onnx-comparison.json"
    if onnx_file.is_file():
        onnx = json.loads(onnx_file.read_text(encoding="utf-8"))
        onnx_folder = Path(onnx["dataset"])
        onnx_result = Path(onnx["results_file"])
        if (not onnx_folder.resolve().is_relative_to((root / "eval/benchmarks/paddle-onnx-comparison").resolve())
            or not onnx_result.resolve().is_relative_to(onnx_folder.resolve())):
            raise ValueError("ONNX control export must remain inside its synthetic corpus")
        files += [onnx_folder / "dataset.json", onnx_folder / "resources.json", onnx_result]
        files += list(onnx_result.parent.glob("page_*.txt"))
        files += [Path(".onnx-models/PP-OCRv6_medium/conversion.json")]
    deployment_file = root / "publication/deployment-comparison.json"
    if deployment_file.is_file():
        deployment = json.loads(deployment_file.read_text(encoding="utf-8"))
        deployment_folder = Path(deployment["dataset"])
        if not deployment_folder.resolve().is_relative_to((root / "eval/benchmarks/deployment-onnx-12").resolve()):
            raise ValueError("Deployment aggregate export is outside its benchmark directory")
        files += [deployment_folder / name for name in ("dataset.json", "publication-stats.md", "publication-stats.json", "page-stats.csv")]
        files += list(deployment_folder.glob("*-resources.json"))
    fullbook_file = root / "publication/fullbook-comparison.json"
    if fullbook_file.is_file():
        fullbook = json.loads(fullbook_file.read_text(encoding="utf-8"))
        fullbook_folder = Path(fullbook["dataset"])
        if not fullbook_folder.resolve().is_relative_to((root / "eval/benchmarks/fullbook-onnx-114").resolve()):
            raise ValueError("Full-book aggregate export is outside its benchmark directory")
        files += [fullbook_folder / name for name in ("dataset.json", "publication-stats.md", "publication-stats.json", "page-stats.csv")]
        files += list(fullbook_folder.glob("*-resources.json"))
        snapshot = fullbook_folder / "host-snapshot.json"
        if snapshot.is_file():
            files.append(snapshot)
    status_file = root / "publication/completion-status.json"
    cpu_file = root / "publication/cpu-four-comparison.json"
    if cpu_file.is_file():
        cpu = json.loads(cpu_file.read_text(encoding="utf-8"))
        cpu_folder = Path(cpu["dataset"])
        if not cpu.get("completed") or not cpu_folder.resolve().is_relative_to((root / "eval/benchmarks/cpu-four-114").resolve()):
            raise ValueError("Four-engine export needs a completed run inside its benchmark directory")
        files += [cpu_folder / name for name in ("dataset.json", "publication-stats.md", "publication-stats.json", "page-stats.csv")]
        files += list(cpu_folder.glob("*-resources.json"))
        files.append(Path(".ocr-models/PP-OCRv6_tiny/provenance.json"))
        cpu_status = json.loads((root / "publication/cpu-four-status.json").read_text(encoding="utf-8"))
        probes = Path(cpu_status["probes"])
        if not probes.resolve().is_relative_to((root / "eval/cpu-four-probes").resolve()):
            raise ValueError("Four-engine probe export is outside its control directory")
        files += [probes / f"{engine}.json" for engine in ("indra", "tesseract", "easyocr", "paddleocr_tiny")]
    if status_file.exists():
        status = json.loads(status_file.read_text(encoding="utf-8"))
        if status.get("completed"):
            public = Path(status["dataset"])
            if not public.resolve().is_relative_to((root / "eval/benchmarks").resolve()):
                raise ValueError("Public aggregate export must stay in the benchmark directory")
            files += [public / name for name in ("dataset.json", "publication-stats.md", "publication-stats.json", "page-stats.csv")]
            files += list(public.glob("publication-*.resources.json"))
    files = list(dict.fromkeys(files))
    packages = {}
    for name in ("indra-ocr", "paddleocr", "paddlepaddle", "paddlex", "numpy", "onnxruntime",
                 "opencv-python-headless", "opencv-contrib-python", "Pillow", "pypdfium2", "rapidfuzz", "psutil"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed in exporter environment"
    environment = {"captured_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "packages": packages, "opencv_runtime": cv2.__version__, "opencv_default_threads": cv2.getNumThreads(),
        "scope": "currently observed environment, not a reconstructed historical full-run environment",
        "note": "both OpenCV distributions are installed; actual cv2 runtime is recorded separately"}
    hashes = {}
    for relative in files:
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Invalid export input: {relative}")
        hashes[relative.as_posix()] = sha256(path)
    destination = root / "publication"
    destination.mkdir(exist_ok=True)
    archive = destination / ("ocr-comparison-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative in files:
            bundle.write(root / relative, relative.as_posix())
        bundle.writestr("environment-snapshot.json", json.dumps(environment, indent=2))
        bundle.writestr("file-hashes.json", json.dumps(hashes, indent=2))
        bundle.writestr("CONTENTS.md", "# Review package\n\nCode, aggregate book metrics and original synthetic control images/text only.\nNo source book PDF, book page images, extracted book text, book predictions, model weights, credentials or font files are included.\nThe article remains a draft. Public annotated-document results are pending download access.\nThe book cannot be reproduced from this archive alone: its exact input PDF is not distributed.\n")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Archive integrity check failed.")
    archive.with_suffix(".zip.sha256").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    print(archive)
    print(f"Verified {len(files)} explicitly selected files; source book and its text/images excluded.")


if __name__ == "__main__":
    main()
