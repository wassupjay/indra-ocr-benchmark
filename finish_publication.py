"""One unattended annotated-data run, evidence audit, article and archive build."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmark import ENGINES, load_dataset, save_json, summarize
from evaluate import sha256
from prepare_public import fetch_source, prepare
from profile_ocr import monitor
from publication_report import build


def completed_run(folder, engine, threads, repeats, paddle_backend="plain", model_dir=None):
    """Resume only a matching successful configuration, never partial outputs."""
    for path in sorted((folder / engine).glob("*/results.json"), reverse=True)[:1]:
        result = json.loads(path.read_text(encoding="utf-8"))
        if (result.get("dataset_sha256") != sha256(folder / "dataset.json")
            or result.get("runner_sha256") != sha256(Path("benchmark.py"))
            or result.get("hardware", {}).get("threads") != threads
            or len(result.get("pass_seconds", [])) != repeats or result.get("errors")):
            continue
        if engine == "paddleocr":
            metadata = result.get("metadata", {})
            settings = metadata.get("settings", {})
            if paddle_backend == "plain":
                if settings.get("enable_mkldnn") is not False or settings.get("engine") == "onnxruntime":
                    continue
            else:
                if settings.get("engine") != "onnxruntime" or model_dir is None:
                    continue
                valid = True
                for name in ("detector", "recognizer"):
                    runtime = metadata.get("effective_runtime", {}).get(name, {})
                    model_files = list((model_dir / name).glob("inference.*"))
                    if (runtime.get("actual_providers") != ["CPUExecutionProvider"]
                        or runtime.get("actual_intra_op_threads") != threads
                        or runtime.get("actual_inter_op_threads") != 1
                        or runtime.get("denormal_as_zero") != "1"
                        or not (model_dir / name / "inference.onnx").is_file()
                        or any(metadata.get("model_hashes", {}).get(name, {}).get(p.name) != sha256(p) for p in model_files)):
                        valid = False
                if not valid:
                    continue
        return path
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--source", type=Path, help="Already downloaded OmniDocBench folder; otherwise fetch normally over HTTPS")
    parser.add_argument("--timeout", type=float, default=10800, help="Maximum whole-worker seconds per engine")
    parser.add_argument("--paddle-backend", choices=("plain", "onnx-ftz"), default="onnx-ftz")
    parser.add_argument("--paddle-model-dir", type=Path, default=Path(".onnx-models/PP-OCRv6_medium"))
    args = parser.parse_args()
    if args.count < 1 or args.timeout <= 0:
        parser.error("count and timeout must be positive")
    os.chdir(Path(__file__).resolve().parent)
    output = Path("publication")
    output.mkdir(exist_ok=True)
    status = {"started_utc": datetime.now(timezone.utc).isoformat(), "phase": "preparing annotated data", "completed": False}
    state_path = output / "completion-status.json"
    save_json(state_path, status)
    try:
        if args.source:
            source, revision, fetch = args.source, "local snapshot; see annotation hash", False
        else:
            source, revision = fetch_source(Path("eval/public-source/OmniDocBench"))
            fetch = True
        folder = prepare(source, args.count, revision, fetch)
        manifest = load_dataset(folder)
        if len(manifest["pages"]) != args.count:
            raise ValueError(f"Requested {args.count} regions, only {len(manifest['pages'])} eligible; stopping instead of silently shrinking the study")
        status["dataset"] = str(folder)
        save_json(state_path, status)
        for engine in ENGINES:
            status["phase"] = f"running {engine}"
            save_json(state_path, status)
            existing = completed_run(folder, engine, 4, 3, args.paddle_backend, args.paddle_model_dir)
            if existing:
                print(f"Resuming completed {engine}: {existing}", flush=True)
                continue
            tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            log = folder / f"publication-{engine}-{tag}.log"
            print(f"Running {engine}: {log}", flush=True)
            command = [sys.executable, "-u", "benchmark.py", "--engine", engine,
                       "--dataset", str(folder), "--threads", "4", "--repeats", "3", "--paddle-backend", args.paddle_backend,
                       "--paddle-model-dir", str(args.paddle_model_dir)]
            resources = monitor(command, args.timeout, log)
            resource_path = log.with_suffix(".resources.json")
            save_json(resource_path, {**resources, "command": command,
                "scope": "entire annotated-subset worker: imports, initialization, warmup, all 3 passes, scoring and writes; not inference-only CPU/RAM",
                "dataset_sha256": sha256(folder / "dataset.json")})
            save_json(folder / f"{engine}-resources.json", {**resources, "command": command,
                "scope": "entire worker: imports, initialization, warmup, all passes, scoring and writes; summed RSS may double-count shared pages"})
            if resources["timed_out"] or resources["exit_code"] != 0:
                raise RuntimeError(f"{engine} did not complete; see {log}. Saved successful engines can resume.")
            if not completed_run(folder, engine, 4, 3, args.paddle_backend, args.paddle_model_dir):
                raise RuntimeError(f"{engine} has failed pages or mismatched configuration; see {log}")
        status["phase"] = "auditing and packaging"
        save_json(state_path, status)
        summarize(folder, manifest)
        build(folder)
        status.update(completed=True, phase="annotated case-study follow-up complete", finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(state_path, status)
        subprocess.run([sys.executable, "export_results.py"], check=True)
        print("Complete: publication/ARTICLE.md and publication/index.html. Nothing was published externally.")
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as error:
        status.update(phase="incomplete", error=str(error), finished_utc=datetime.now(timezone.utc).isoformat())
        save_json(state_path, status)
        # Preserve a useful, audited local case study even when external data is unavailable.
        build()
        subprocess.run([sys.executable, "export_results.py"], check=True)
        parser.exit(1, f"Annotated follow-up incomplete: {error}\nLocal case-study package rebuilt. See publication/completion-status.json.\n")


if __name__ == "__main__":
    main()
