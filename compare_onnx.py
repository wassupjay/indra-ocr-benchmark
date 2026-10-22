"""Compare ONNX CPU inference with the measured plain-Paddle control baseline."""

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shutil
import statistics
import sys

from benchmark import load_dataset, save_json
from evaluate import sha256
from profile_ocr import monitor


def validate_comparison(result, baseline, conversion, manifest):
    if baseline["source_dataset_sha256"] != manifest["source_dataset_sha256"]:
        raise ValueError("Comparison inputs do not match baseline")
    for name in ("detector", "recognizer"):
        if conversion["models"][name]["source_sha256"] != baseline["metadata"]["model_hashes"][name]:
            raise ValueError("Converted source weights do not match baseline")
        runtime = result["metadata"]["effective_runtime"][name]
        if runtime.get("actual_providers") != ["CPUExecutionProvider"] or runtime.get("actual_intra_op_threads") != 4:
            raise ValueError("ONNX did not use the requested CPU provider/four threads")
    details = result.get("pass_details", [])
    expected_ids = [p["page"] for p in manifest["pages"]]
    if result["errors"] or len(details) != 3 or any([p["page"] for p in t["pages"]] != expected_ids for t in details):
        raise ValueError("Incomplete or failed ONNX measurement")
    return {"all_passes_zero_edits": all(t["overall"]["word_errors"] == 0 and t["overall"]["character_errors"] == 0 for t in details),
        "normalized_outputs_match_baseline": all(
            [p["normalized_text_sha256"] for p in t["pages"]] ==
            [p["normalized_text_sha256"] for p in baseline["raw_pass_details"][0]["pages"]] for t in details)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flush-denormals", action="store_true")
    args = parser.parse_args()
    source = Path("eval/benchmarks/synthetic-control-v1")
    manifest = load_dataset(source)
    baseline = json.loads(Path("publication/paddle-verification.json").read_text(encoding="utf-8"))
    baseline_result = Path(baseline["results_file"])
    if sha256(baseline_result) != baseline["results_sha256"]:
        raise ValueError("Baseline result changed")
    baseline["raw_pass_details"] = json.loads(baseline_result.read_text(encoding="utf-8"))["pass_details"]
    models = Path(".onnx-models/PP-OCRv6_medium")
    conversion = json.loads((models / "conversion.json").read_text(encoding="utf-8"))
    for name in ("detector", "recognizer"):
        if sha256(models / name / "inference.onnx") != conversion["models"][name]["onnx_sha256"]:
            raise ValueError("Converted model changed")
    folder = Path("eval/benchmarks/paddle-onnx-comparison") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    for page in manifest["pages"]:
        for kind in ("image", "reference"):
            shutil.copyfile(source / page[kind], folder / page[kind])
    manifest = {**manifest, "source_dataset_sha256": sha256(source / "dataset.json")}
    save_json(folder / "dataset.json", manifest)
    command = [sys.executable, "-u", "benchmark.py", "--engine", "paddleocr", "--dataset", str(folder),
        "--threads", "4", "--repeats", "3", "--paddle-backend", "onnx-ftz" if args.flush_denormals else "onnx", "--paddle-model-dir", str(models)]
    print(f"ONNX comparison: {folder}", flush=True)
    resources = monitor(command, 3600, folder / "onnx.log")
    resources["scope"] = "whole worker including imports, loading, warmup, all three passes, scoring and writes; summed RSS may double-count shared memory"
    save_json(folder / "resources.json", resources)
    files = sorted((folder / "paddleocr").glob("*/results.json"))
    if resources["exit_code"] != 0 or resources["timed_out"] or not files:
        raise RuntimeError(f"ONNX run failed or timed out; see {folder / 'onnx.log'}")
    result_path = files[-1]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checks = validate_comparison(result, baseline, conversion, manifest)
    if args.flush_denormals and any(result["metadata"]["effective_runtime"][name].get("denormal_as_zero") != "1" for name in ("detector", "recognizer")):
        raise ValueError("Requested floating-point setting was not applied to both sessions")
    summary = {**checks, "comparison_completed": True, "flush_denormals": args.flush_denormals, "dataset": str(folder),
        "dataset_sha256": sha256(folder / "dataset.json"), "baseline_results_sha256": baseline["results_sha256"],
        "results_file": str(result_path), "results_sha256": sha256(result_path),
        "conversion": conversion, "baseline_median_pass_seconds": baseline["median_pass_seconds"],
        "onnx_median_pass_seconds": result["median_pass_seconds"], "onnx_pass_seconds": result["pass_seconds"],
        "baseline_control_images_per_minute": baseline["pages_per_minute"], "onnx_control_images_per_minute": result["pages_per_minute"],
        "observed_speed_ratio": baseline["median_pass_seconds"] / result["median_pass_seconds"],
        "onnx_pass_time_cv_percent": statistics.stdev(result["pass_seconds"]) / statistics.mean(result["pass_seconds"]) * 100,
        "resources": resources, "metadata": result["metadata"],
        "limitation": "three synthetic variants of one original passage; separate sequential runs, not interleaved; a control-set runtime comparison, not universal accuracy/speed"}
    save_json(folder / "comparison.json", summary)
    save_json(Path("publication/onnx-comparison.json"), summary)
    save_json(Path("publication/onnx-comparison-status.json"), {"phase": "completed", **checks, "comparison_completed": True, "conversion_completed": True})
    lines = ["# Paddle versus ONNX Runtime: CPU control comparison", "",
        "Same source weights, image hashes, preprocessing, one excluded warmup and three measured passes. Four requested inference threads; ONNX actual CPU provider/thread settings verified.", "",
        f"ONNX session.set_denormal_as_zero: {int(args.flush_denormals)}. This floating-point setting can affect numerical outputs; quality and text-hash comparisons below are required.", "",
        "| Runtime | Median three-image pass (s) | Control images/min |", "|---|---:|---:|",
        f"| Plain Paddle | {baseline['median_pass_seconds']:.3f} | {baseline['pages_per_minute']:.3f} |",
        f"| ONNX Runtime | {result['median_pass_seconds']:.3f} | {result['pages_per_minute']:.3f} |", "",
        f"Observed time ratio on this control: {summary['observed_speed_ratio']:.2f}x. ONNX pass-time CV: {summary['onnx_pass_time_cv_percent']:.2f}%.",
        f"Zero WER/CER on every ONNX pass: {checks['all_passes_zero_edits']}. All normalized outputs match baseline: {checks['normalized_outputs_match_baseline']}.",
        f"Sampled ONNX peak tree RSS: {resources['sampled_peak_tree_rss_mib']:.1f} MiB. Whole-worker mean busy cores: {resources['mean_busy_cpu_cores_whole_worker']:.2f}.", "",
        summary["limitation"], "", resources["scope"], "",
        "Conversion used the same Paddle2ONNX native export API called by its Python wrapper, with the exact matching isolated Paddle build. FP16/quantization disabled. Full ONNX model checks passed; source and converted model hashes are recorded."]
    Path("publication/onnx-comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    if not all(checks.values()):
        raise RuntimeError("Completed comparison has quality differences; inspect report before promotion")


if __name__ == "__main__":
    main()
