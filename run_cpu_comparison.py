"""Set up and run Indra, Tesseract, EasyOCR and PP-OCRv6 Tiny on CPU."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

from benchmark import configured_engine, load_dataset, save_json
from evaluate import compare, normalize, sha256
from profile_ocr import monitor
from publication_report import audit_dataset, table_image


ROOT = Path(__file__).resolve().parent
ENGINES = ("indra", "tesseract", "easyocr", "paddleocr_tiny")
LABELS = {"indra": "Indra", "tesseract": "Tesseract 5", "easyocr": "EasyOCR", "paddleocr_tiny": "PaddleOCR Tiny"}
MODELS = ROOT / ".ocr-models/PP-OCRv6_tiny"
EASY_PYTHON = ROOT / ".easyocr-venv/Scripts/python.exe"
BASE_PYTHON = ROOT / ".benchmark-venv/Scripts/python.exe"
SOURCE = ROOT / "eval/benchmarks/f960d85713a3-200dpi-all"
STATE = ROOT / "publication/cpu-four-status.json"


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.load(response)


def prepare_tiny():
    provenance_path = MODELS / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        for component, item in provenance["models"].items():
            if component not in ("detector", "recognizer"):
                raise ValueError("Unknown Tiny model component")
            if set(item["files"]) != {"inference.onnx", "inference.yml"}:
                raise ValueError("Tiny provenance must hash both model and config")
            for name, digest in item["files"].items():
                if name not in ("inference.onnx", "inference.yml") or sha256(MODELS / component / name) != digest:
                    raise ValueError("Cached Tiny model changed")
        if set(provenance["models"]) != {"detector", "recognizer"}:
            raise ValueError("Tiny provenance is incomplete")
        return
    provenance = {"source": "official PaddlePaddle ONNX packages; no local conversion or quantization",
                  "models": {}}
    for component, suffix in (("detector", "det"), ("recognizer", "rec")):
        repo = f"PaddlePaddle/PP-OCRv6_tiny_{suffix}_onnx"
        info = fetch_json(f"https://huggingface.co/api/models/{repo}")
        revision = info["sha"]
        if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
            raise ValueError("Invalid Hugging Face model revision")
        destination = MODELS / component
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("inference.onnx", "inference.yml"):
            path = destination / name
            part = path.with_name("." + path.name + ".part")
            with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", timeout=60) as response:
                with part.open("wb") as output:
                    shutil.copyfileobj(response, output)
            part.replace(path)
        import yaml
        config = yaml.safe_load((destination / "inference.yml").read_text(encoding="utf-8"))
        if config.get("Global", {}).get("model_name") != f"PP-OCRv6_tiny_{suffix}":
            raise ValueError("Downloaded config is not the requested Tiny model")
        provenance["models"][component] = {"repo": repo, "revision": revision,
            "files": {name: sha256(destination / name) for name in ("inference.onnx", "inference.yml")}}
    save_json(provenance_path, provenance)


def setup():
    # Fail before modifying environments if this execution context cannot download.
    fetch_json("https://pypi.org/pypi/easyocr/1.7.2/json")
    if not EASY_PYTHON.exists():
        subprocess.run([str(BASE_PYTHON), "-m", "venv", "--without-pip", str(EASY_PYTHON.parents[1])], check=True)
    site = EASY_PYTHON.parents[1] / "Lib/site-packages"
    site.mkdir(parents=True, exist_ok=True)
    (site / "benchmark-dependencies.pth").write_text(str(BASE_PYTHON.parents[1] / "Lib/site-packages") + "\n", encoding="utf-8")
    temporary = ROOT / ".cpu-comparison-temp"
    temporary.mkdir(exist_ok=True)
    env = {**os.environ, "TEMP": str(temporary), "TMP": str(temporary),
           "PIP_CACHE_DIR": str(ROOT / ".package-cache"), "PIP_NO_INDEX": "0"}
    pip = [str(BASE_PYTHON), "-m", "pip", "--python", str(EASY_PYTHON), "install", "--timeout", "30", "--retries", "0"]
    subprocess.run(pip + ["--index-url", "https://download.pytorch.org/whl/cpu", "torch==2.8.0+cpu", "torchvision==0.23.0+cpu"], env=env, check=True)
    subprocess.run(pip + ["--index-url", "https://pypi.org/simple", "-c", "benchmark-requirements.txt", "easyocr==1.7.2"], env=env, check=True)
    model_dir = ROOT / ".ocr-models/easyocr"
    model_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(EASY_PYTHON), "-c",
        "import sys,easyocr,torch; assert torch.version.cuda is None; "
        "easyocr.Reader(['en'],gpu=False,model_storage_directory=sys.argv[1],download_enabled=True,verbose=False,quantize=True)",
        str(model_dir)], env=env, check=True)
    prepare_tiny()


def missing_assets():
    required = [EASY_PYTHON, MODELS / "provenance.json"]
    required += [ROOT / ".ocr-models/easyocr" / name for name in ("craft_mlt_25k.pth", "english_g2.pth")]
    required += [MODELS / name / file for name in ("detector", "recognizer") for file in ("inference.onnx", "inference.yml")]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if EASY_PYTHON.is_file():
        try:
            result = subprocess.run([str(EASY_PYTHON), "-c",
                "import easyocr,torch,torchvision; assert easyocr.__version__=='1.7.2'; "
                "assert torch.__version__.startswith('2.8.0+cpu') and torch.version.cuda is None; "
                "assert torchvision.__version__.startswith('0.23.0+cpu')"],
                capture_output=True, timeout=90)
            if result.returncode != 0:
                missing.append("EasyOCR/PyTorch CPU dependencies")
        except (OSError, subprocess.TimeoutExpired):
            missing.append("EasyOCR/PyTorch CPU dependencies")
    return missing


def probe(engine, output):
    os.environ.update(OMP_NUM_THREADS="4", OMP_THREAD_LIMIT="4", PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True")
    start = time.perf_counter()
    recognize, metadata = configured_engine(engine, 4, "onnx-ftz", MODELS)
    initialization = time.perf_counter() - start
    folder = ROOT / "eval/benchmarks/synthetic-control-v1"
    prediction = recognize(folder / "page_003.png")
    score = compare(normalize((folder / "page_003.txt").read_text(encoding="utf-8"), []), normalize(prediction, []))
    if engine == "easyocr":
        if (metadata["device"] != "cpu" or metadata["torch_intra_op_threads"] != 4
            or metadata["torch_inter_op_threads"] != 1 or not metadata["torch"].startswith("2.8.0+cpu")):
            raise ValueError("Unexpected EasyOCR CPU runtime")
    if engine == "paddleocr_tiny":
        provenance = json.loads((MODELS / "provenance.json").read_text(encoding="utf-8"))
        for component in ("detector", "recognizer"):
            runtime = metadata["effective_runtime"][component]
            if (runtime.get("actual_providers") != ["CPUExecutionProvider"]
                or runtime.get("actual_intra_op_threads") != 4 or runtime.get("actual_inter_op_threads") != 1
                or runtime.get("denormal_as_zero") != "1"
                or metadata["model_hashes"][component] != provenance["models"][component]["files"]):
                raise ValueError("Tiny runtime or model hashes do not match the prepared CPU configuration")
    save_json(output, {"engine": engine, "metadata": metadata, "initialization_seconds": initialization,
                      "control_score": {k: v for k, v in score.items() if k != "changes"},
                      "scope": "one known-text control image; a functional probe, not full-book results"})


def matching_result(folder, engine, metadata):
    files = sorted((folder / engine).glob("*/results.json"))
    if not files:
        return None
    path = files[-1]
    result = json.loads(path.read_text(encoding="utf-8"))
    resource_path = folder / f"{engine}-resources.json"
    if not resource_path.is_file():
        return None
    resources = json.loads(resource_path.read_text(encoding="utf-8"))
    if (resources.get("timed_out") or resources.get("exit_code") != 0 or result.get("engine") != engine
        or result.get("dataset_sha256") != sha256(folder / "dataset.json")
        or result.get("runner_sha256") != sha256(ROOT / "benchmark.py")
        or result.get("metadata") != metadata or result.get("hardware", {}).get("threads") != 4
        or len(result.get("pass_seconds", [])) != 3 or result.get("errors")):
        return None
    return path


def prepare_dataset():
    original = load_dataset(SOURCE)
    if STATE.exists():
        previous = json.loads(STATE.read_text(encoding="utf-8"))
        if previous.get("dataset"):
            candidate = Path(previous["dataset"]).resolve()
            if not candidate.is_relative_to(ROOT / "eval/benchmarks/cpu-four-114"):
                raise ValueError("Resume dataset must stay in the four-engine benchmark directory")
            manifest = load_dataset(candidate)
            if (manifest.get("source_dataset_sha256") != sha256(SOURCE / "dataset.json")
                or manifest.get("engines") != list(ENGINES) or manifest["pages"] != original["pages"]):
                raise ValueError("Resume dataset settings changed")
            return candidate
    folder = ROOT / "eval/benchmarks/cpu-four-114" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    for item in original["pages"]:
        for name in ("image", "reference"):
            shutil.copyfile(SOURCE / item[name], folder / item[name])
    save_json(folder / "dataset.json", {**original, "engines": list(ENGINES),
        "source_dataset_sha256": sha256(SOURCE / "dataset.json"),
        "page_selection": "all source pages in original order; identical saved PNG and reference bytes"})
    return folder


def write_report(folder):
    manifest = load_dataset(folder)
    count = len(manifest["pages"])
    audit = audit_dataset(folder)
    stats = json.loads((folder / "publication-stats.json").read_text(encoding="utf-8"))
    resources = {e: json.loads((folder / f"{e}-resources.json").read_text(encoding="utf-8")) for e in ENGINES}
    report = {"completed": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "dataset": str(folder.relative_to(ROOT)), "audit": audit,
              "statistics": stats, "resources": resources,
              "reference_status": manifest["reference_status"]}
    resolution = f"{manifest['dpi']}-DPI" if "dpi" in manifest else "original-resolution"
    lines = ["# Four-engine CPU comparison", "", report["reference_status"], "",
        f"Same {count} saved {resolution} RGB page images. Four requested inference threads, one first-image warmup, three measured passes. Separate sequential engine workers on the same PC.", "",
        "| Engine | WER | CER | Pages/min | Engine init (s) | Peak tree RSS (MiB) | Pass CV | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    rows = []
    for engine in ENGINES:
        model = stats["models"][engine]
        ram = resources[engine]["sampled_peak_tree_rss_mib"]
        lines.append(f"| {LABELS[engine]} | {model['wer']:.3%} | {model['cer']:.3%} | {model['pages_per_minute']:.2f} | {model['initialization_seconds']:.3f} | {ram:.1f} | {model['pass_time_cv_percent']:.2f}% | {model['failures_all_passes']} |")
        rows.append([LABELS[engine], f"{model['wer']:.3%}", f"{model['cer']:.3%}", f"{model['pages_per_minute']:.2f}", f"{ram:.1f}"])
    lines += ["", "Indra is forced into OCR mode. Tesseract uses English/OEM1/PSM3 and starts one CLI process per page. EasyOCR uses English CRAFT + english_g2, CPU, greedy decoding, batch 1, zero data-loader workers and its requested default dynamic quantization; actual quantized module types are recorded. Paddle uses the official PP-OCRv6 Tiny ONNX detector/recognizer with CPUExecutionProvider, four intra-op/one inter-op threads and denormal-as-zero enabled. Tiny is not the historical medium model.",
        "", "Throughput counts serial image-to-text calls, including file decode and Tesseract CLI startup. Loading, warmup, scoring and writes are excluded. Engine init is measured within each worker and includes engine imports and loading; it is not a guarantee of true cold-start latency. All engines retain scores and normalized output hashes for every pass. First-pass predictions were rescored from saved text.",
        "", "CPU/RAM are sampled every 100 ms over the entire worker, including imports, loading, warmup, inference, scoring and writes. Summed RSS can double-count shared pages; sampling can miss short peaks or short-lived child CPU work. Resource and throughput scopes differ. Historical/subset timings are separate experiments. This one-book comparison does not establish a universal winner."]
    (ROOT / "publication/cpu-four-comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "publication/cpu-four-page-stats.csv").write_bytes((folder / "page-stats.csv").read_bytes())
    table_image(ROOT / "publication/cpu-four-results.png", f"{count} pages: four OCR engines on the same CPU",
        ["Engine", "WER", "CER", "Pages/min", "Peak MiB"], rows,
        ["Unverified PDF-text reference; scores measure text agreement, not verified accuracy.",
         "Three measured passes per engine; one warmup excluded. CPU only, four requested threads.",
         "Peak RAM covers whole workers; throughput counts image-to-text calls only."])
    save_json(ROOT / "publication/cpu-four-comparison.json", report)
    print("\n".join(lines), flush=True)


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", action="store_true", help="Install isolated EasyOCR CPU dependencies, download models, then run all four")
    parser.add_argument("--check", action="store_true", help="Only check whether the required local assets exist")
    parser.add_argument("--probe", choices=ENGINES, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.probe:
        if args.output is None:
            parser.error("--probe requires --output")
        probe(args.probe, args.output)
        return
    if args.check:
        missing = missing_assets()
        if missing:
            parser.exit(1, "Missing assets: " + ", ".join(missing) + "\n")
        prepare_tiny()
        print("Assets present; this check alone does not run inference or establish completed results.")
        return
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    state.pop("error", None)
    state.pop("checked_utc", None)
    state.update(completed=False, phase="checking dependencies", engines=list(ENGINES))
    STATE.parent.mkdir(exist_ok=True)
    save_json(STATE, state)
    try:
        if args.setup and missing_assets():
            state["phase"] = "downloading CPU dependencies and models"
            save_json(STATE, state)
            setup()
        missing = missing_assets()
        if missing:
            raise RuntimeError("Missing assets: " + ", ".join(missing) + ". Run with --setup in a terminal with download access.")
        prepare_tiny()  # Validate cached provenance, without downloading.
        probe_folder = ROOT / "eval/cpu-four-probes" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        probe_folder.mkdir(parents=True)
        metadata = {}
        for engine in ENGINES:
            python = EASY_PYTHON if engine == "easyocr" else BASE_PYTHON
            output = probe_folder / f"{engine}.json"
            state["phase"] = f"probing {engine}"
            save_json(STATE, state)
            resources = monitor([str(python), "-u", __file__, "--probe", engine, "--output", str(output)], 600, output.with_suffix(".log"))
            if resources["timed_out"] or resources["exit_code"] != 0 or not output.exists():
                raise RuntimeError(f"{engine} functional probe failed; see {output.with_suffix('.log')}")
            metadata[engine] = json.loads(output.read_text(encoding="utf-8"))["metadata"]
        folder = prepare_dataset()
        state.update(dataset=str(folder.relative_to(ROOT)), probes=str(probe_folder.relative_to(ROOT)))
        save_json(STATE, state)
        for engine in ENGINES:
            state["phase"] = f"running {engine}"
            save_json(STATE, state)
            if matching_result(folder, engine, metadata[engine]):
                print(f"Resume verified completed run: {engine}", flush=True)
                continue
            python = EASY_PYTHON if engine == "easyocr" else BASE_PYTHON
            command = [str(python), "-u", "benchmark.py", "--dataset", str(folder), "--engine", engine,
                "--threads", "4", "--repeats", "3", "--paddle-backend", "onnx-ftz", "--paddle-model-dir", str(MODELS)]
            print(f"Running {LABELS[engine]}; log: {folder / (engine + '.log')}", flush=True)
            resources = monitor(command, 10800, folder / f"{engine}.log")
            resources["scope"] = "whole engine worker including imports, initialization, warmup, all passes, scoring and writes; summed RSS can double-count shared memory"
            save_json(folder / f"{engine}-resources.json", resources)
            if resources["timed_out"] or resources["exit_code"] != 0 or not matching_result(folder, engine, metadata[engine]):
                raise RuntimeError(f"{engine} incomplete or changed configuration; see {folder / (engine + '.log')}")
        state["phase"] = "auditing"
        save_json(STATE, state)
        write_report(folder)
        state.pop("error", None)
        state.update(completed=True, phase="completed", finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(STATE, state)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        state.update(completed=False, phase="incomplete", error=str(error), checked_utc=datetime.now(timezone.utc).isoformat())
        save_json(STATE, state)
        parser.exit(1, f"Four-engine comparison incomplete: {error}\nNo new four-engine publication results have been promoted.\n")


if __name__ == "__main__":
    main()
