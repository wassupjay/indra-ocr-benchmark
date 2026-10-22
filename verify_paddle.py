"""Repeatable local Paddle CPU control measurement; preserves book results."""

from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from benchmark import load_dataset, save_json
from evaluate import sha256
from profile_ocr import monitor


def assess(result, resources, expected_pages):
    details = result.get("pass_details", [])
    runtime = result["metadata"].get("effective_runtime", {})
    ids = [p["page"] for p in expected_pages]
    complete = (bool(ids) and not resources["timed_out"] and resources["exit_code"] == 0
        and not result["errors"] and len(details) == 3
        and len(result["pass_seconds"]) == 3
        and all(math.isfinite(s) and s > 0 for s in result["pass_seconds"])
        and all([p["page"] for p in trial["pages"]] == ids for trial in details))
    accurate = complete and all(t["overall"]["word_errors"] == 0 and t["overall"]["character_errors"] == 0 for t in details)
    stable_text = complete and all(
        [p["normalized_text_sha256"] for p in trial["pages"]]
        == [p["normalized_text_sha256"] for p in details[0]["pages"]] for trial in details)
    correct_runtime = all(runtime.get(k, {}).get("run_mode") == "paddle"
        and runtime[k].get("cpu_threads") == 4 for k in ("detector", "recognizer"))
    hashes = result["metadata"].get("model_hashes", {})
    weights_recorded = all("inference.pdiparams" in hashes.get(k, {}) for k in ("detector", "recognizer"))
    return {"configuration_verified_on_control": bool(complete and accurate and stable_text and correct_runtime and weights_recorded),
        "repeated_measurement_complete": bool(complete), "zero_edits_all_passes": bool(accurate),
        "identical_normalized_text_all_passes": bool(stable_text),
        "actual_runtime_matches_request": bool(correct_runtime), "model_weights_hashed": bool(weights_recorded)}


def main():
    source = Path("eval/benchmarks/synthetic-control-v1")
    original = load_dataset(source)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = Path("eval/benchmarks/paddle-runtime-verification") / tag
    folder.mkdir(parents=True)
    for page in original["pages"]:
        for kind in ("image", "reference"):
            shutil.copyfile(source / page[kind], folder / page[kind])
    manifest = {**original, "source_dataset_sha256": sha256(source / "dataset.json"),
        "verification_protocol": "unchanged 3 known-text images; one first-image warmup; 3 serial measured passes; plain CPU, four requested threads; numerical quality and normalized text hashes on every pass; whole-worker resources"}
    save_json(folder / "dataset.json", manifest)
    load_dataset(folder)
    command = [sys.executable, "-u", "benchmark.py", "--engine", "paddleocr", "--dataset", str(folder),
               "--threads", "4", "--repeats", "3", "--paddle-backend", "plain"]
    save_json(folder / "verification-status.json", {"phase": "running", "command": command, "timeout_seconds": 3600})
    print(f"Verification folder: {folder}", flush=True)
    resources = monitor(command, 3600, folder / "paddleocr.log")
    resources["scope"] = "whole worker including imports, initialization, one warmup, all 3 measured passes, scoring and writes; summed RSS can double-count shared pages; CPU/RAM are not inference-only"
    save_json(folder / "resources.json", resources)
    files = sorted((folder / "paddleocr").glob("*/results.json"))
    if resources["timed_out"] or resources["exit_code"] != 0 or not files:
        save_json(folder / "verification-status.json", {"phase": "incomplete", "resources": resources})
        raise RuntimeError(f"Verification did not finish; see {folder / 'paddleocr.log'}")
    result_path = files[-1]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checks = assess(result, resources, manifest["pages"])
    passes = result["pass_seconds"]
    summary = {**checks, "scope": "three synthetic variants of one 124-word passage; configuration check, not a diverse accuracy benchmark or optimized full-book speed",
        "dataset": str(folder), "dataset_sha256": sha256(folder / "dataset.json"),
        "source_dataset_sha256": manifest["source_dataset_sha256"],
        "results_file": str(result_path), "results_sha256": sha256(result_path),
        "images": len(manifest["pages"]), "measured_passes": len(passes),
        "pass_seconds": passes, "median_pass_seconds": result["median_pass_seconds"],
        "pages_per_minute": result["pages_per_minute"],
        "pass_time_cv_percent": statistics.stdev(passes) / statistics.mean(passes) * 100,
        "quality_each_pass": [t["overall"] for t in result["pass_details"]],
        "initialization_seconds": result["initialization_seconds"], "warmup_seconds": result["warmup_seconds"],
        "metadata": result["metadata"], "hardware": result["hardware"], "resources": resources,
        "alternate_runtime_tested": False,
        "onnx_converter_installed": importlib.util.find_spec("paddle2onnx") is not None,
        "optimized_full_book_speed_established": False}
    save_json(folder / "verification-status.json", {"phase": "completed", **checks})
    save_json(folder / "verification.json", summary)
    target = Path("publication")
    target.mkdir(exist_ok=True)
    save_json(target / "paddle-verification.json", summary)
    lines = ["# Repeated Paddle CPU configuration check", "", f"Configuration verified on control: **{checks['configuration_verified_on_control']}**.", "",
        "Unchanged PP-OCRv6 medium weights, three original synthetic images, four CPU threads, plain Paddle backend. One first-image warmup excluded from throughput. Three measured serial passes.", "",
        "| Pass | Image-to-text seconds | WER | CER |", "|---|---:|---:|---:|"]
    for i, trial in enumerate(result["pass_details"], 1):
        lines.append(f"| {i} | {trial['seconds']:.3f} | {trial['overall']['wer']:.3%} | {trial['overall']['cer']:.3%} |")
    lines += ["", f"Median: {result['median_pass_seconds']:.3f} s per three-image pass; {result['pages_per_minute']:.3f} control images/min. Pass-time CV: {summary['pass_time_cv_percent']:.2f}%.",
        f"Peak sampled process-tree RSS: {resources['sampled_peak_tree_rss_mib']:.1f} MiB. Mean busy CPU cores over the whole worker: {resources['mean_busy_cpu_cores_whole_worker']:.2f}.",
        "", resources["scope"], "", summary["scope"],
        "", "Detector/recognizer weight hashes and effective runtime settings are in paddle-verification.json. Original book results were preserved. The ONNX comparison and optimal Paddle speed have not been established."]
    (target / "paddle-verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "publication_report.py"], check=True)
    subprocess.run([sys.executable, "export_results.py"], check=True)
    print("\n".join(lines), flush=True)
    if not checks["configuration_verified_on_control"]:
        raise RuntimeError("Measurements finished but one or more verification checks failed")


if __name__ == "__main__":
    main()
