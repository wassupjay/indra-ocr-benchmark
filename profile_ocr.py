"""Bounded CPU/RAM diagnostic on a shared page; does not replace full-run timings."""

import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from benchmark import ENGINES, load_engine, save_json
from evaluate import compare, normalize, sha256


def worker(engine, dataset, page_number, output, det_max_side=None, legacy_ir=False, skip_warmup=False, threads=4, no_mkldnn=False):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_THREAD_LIMIT"] = str(threads)
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    start = time.perf_counter()
    overrides = {}
    if det_max_side:
        overrides.update(text_det_limit_side_len=det_max_side, text_det_limit_type="max")
    if legacy_ir or no_mkldnn:
        overrides["engine_config"] = {"paddle_static": {"enable_new_ir": not legacy_ir, "cpu_threads": threads, "run_mode": "paddle" if no_mkldnn else "mkldnn"}}
        overrides["enable_mkldnn"] = not no_mkldnn
    recognize, metadata = load_engine(engine, threads, overrides)
    initialization = time.perf_counter() - start
    stage_seconds, shapes = {}, {}
    if engine == "paddleocr":
        # Read the pinned PaddleX runtime objects; do not change backend settings.
        model = inspect.getclosurevars(recognize).nonlocals["model"]
        pipeline = model.paddlex_pipeline._pipeline
        metadata["effective_runtime"] = {}
        for name, component in (("detector", pipeline.text_det_model), ("recognizer", pipeline.text_rec_model)):
            runner = component.runner
            metadata["effective_runtime"][name] = dict(runner._config)

            def timed(*args, _name=name, _runner=runner, **kwargs):
                data = kwargs.get("x")
                if data is not None:
                    shapes[_name] = [list(item.shape) for item in data]
                partial["active_stage"] = _name
                partial["last_stage_input_shapes"] = dict(shapes)
                save_json(output, partial)
                begin = time.perf_counter()
                try:
                    return _runner(*args, **kwargs)
                finally:
                    stage_seconds[_name] = stage_seconds.get(_name, 0) + time.perf_counter() - begin
                    partial["stage_inference_seconds"] = dict(stage_seconds)
                    save_json(output, partial)

            component.runner = timed
    partial = {"engine": engine, "metadata": metadata, "sample_pdf_page": page_number,
               "dataset_sha256": sha256(dataset / "dataset.json"),
               "input_image_sha256": sha256(dataset / f"page_{page_number:03}.png"),
               "initialization_seconds": initialization, "warmup_performed": not skip_warmup, "phase": "cover warmup"}
    save_json(output, partial)
    warmup = 0
    if not skip_warmup:
        warmup = time.perf_counter()
        recognize(dataset / "page_001.png")
        warmup = time.perf_counter() - warmup
    stage_seconds.clear()
    partial.update({"phase": "measured page", "warmup_cover_seconds": warmup})
    save_json(output, partial)
    started = time.perf_counter()
    text = recognize(dataset / f"page_{page_number:03}.png")
    elapsed = time.perf_counter() - started
    output.with_suffix(".txt").write_text(text, encoding="utf-8")
    reference = (dataset / f"page_{page_number:03}.txt").read_text(encoding="utf-8")
    score = compare(normalize(reference, []), normalize(text, []))
    score.pop("changes")
    save_json(output, {"engine": engine, "metadata": metadata, "sample_pdf_page": page_number,
        "dataset_sha256": sha256(dataset / "dataset.json"),
        "input_image_sha256": sha256(dataset / f"page_{page_number:03}.png"),
        "initialization_seconds": initialization, "warmup_cover_seconds": warmup,
        "sample_page_seconds": elapsed, "stage_inference_seconds": stage_seconds,
        "last_stage_input_shapes": shapes, "sample_score": score, "phase": "completed", "warmup_performed": not skip_warmup})


def monitor(command, timeout, log):
    started = time.perf_counter()
    latest_cpu, peak_rss, peak_threads, samples = {}, 0, 0, 0
    minimum_available = psutil.virtual_memory().available
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        root = psutil.Process(process.pid)
        timed_out = False
        while process.poll() is None:
            rss, threads = 0, 0
            try:
                tree = [root, *root.children(recursive=True)]
            except psutil.Error:
                tree = []
            for child in tree:
                try:
                    rss += child.memory_info().rss
                    threads += child.num_threads()
                    cpu = child.cpu_times()
                    latest_cpu[(child.pid, child.create_time())] = cpu.user + cpu.system
                except psutil.Error:
                    continue
            peak_rss = max(peak_rss, rss)
            peak_threads = max(peak_threads, threads)
            minimum_available = min(minimum_available, psutil.virtual_memory().available)
            samples += 1
            if time.perf_counter() - started > timeout:
                timed_out = True
                # These PIDs were created by this diagnostic only.
                for child in reversed(tree):
                    try:
                        child.kill()
                    except psutil.Error:
                        pass
                process.wait(timeout=10)
                break
            time.sleep(.1)
    elapsed = time.perf_counter() - started
    cpu_seconds = sum(latest_cpu.values())
    return {"exit_code": process.returncode, "timed_out": timed_out, "wall_seconds": elapsed,
        "sampled_peak_tree_rss_mib": peak_rss / 1024 ** 2,
        "sampled_tree_cpu_seconds": cpu_seconds, "mean_busy_cpu_cores_whole_worker": cpu_seconds / elapsed,
        "sampled_peak_tree_threads": peak_threads,
        "minimum_system_available_ram_mib": minimum_available / 1024 ** 2,
        "samples": samples, "sampling_seconds": .1,
        "scope": "one worker including imports, model loading, cover warmup and one measured page; summed RSS may double-count shared memory; sampled CPU can miss short-lived child work"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("eval/benchmarks/f960d85713a3-200dpi-all"))
    parser.add_argument("--page", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--worker", choices=ENGINES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--paddle-det-max-side", type=int)
    parser.add_argument("--paddle-legacy-ir", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true", help="Cold single-page diagnostic; do not compare to warmed full-run throughput")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--paddle-no-mkldnn", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if not (args.dataset / f"page_{args.page:03}.png").is_file():
        parser.error("Sample page image does not exist.")
    if args.worker:
        worker(args.worker, args.dataset, args.page, args.output, args.paddle_det_max_side, args.paddle_legacy_ir, args.skip_warmup, args.threads, args.paddle_no_mkldnn)
        return
    folder = args.dataset / "profiles" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    results = {}
    for engine in args.engines:
        target = folder / f"{engine}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", engine,
            "--dataset", str(args.dataset), "--page", str(args.page), "--output", str(target), "--threads", str(args.threads)]
        if args.paddle_no_mkldnn:
            command.append("--paddle-no-mkldnn")
        if args.paddle_det_max_side:
            command += ["--paddle-det-max-side", str(args.paddle_det_max_side)]
        if args.paddle_legacy_ir:
            command.append("--paddle-legacy-ir")
        if args.skip_warmup:
            command.append("--skip-warmup")
        print(f"Profiling {engine}: page {args.page}, timeout {args.timeout}s", flush=True)
        resource = monitor(command, args.timeout, folder / f"{engine}.log")
        result = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {"engine": engine}
        result["resources"] = resource
        save_json(target, result)
        results[engine] = result
        print(f"{engine}: {resource['wall_seconds']:.2f}s worker, {resource['sampled_peak_tree_rss_mib']:.0f} MiB peak sampled RSS, timeout={resource['timed_out']}", flush=True)
    save_json(folder / "profile.json", results)
    write_profile_report(folder, results)
    print(f"Saved {folder / 'profile.json'}")


def write_profile_report(folder, results):
    lines = ["# Resource diagnostic (separate from full-book benchmark)", "",
        "Each worker loads its model, then recognizes one requested image. Normally there is a cover warmup; cold diagnostics explicitly skip it. Check warmup_performed and warmup_cover_seconds in each JSON. This is not full-book peak RAM.", "",
        "| Engine | Sample page (s) | Peak sampled tree RSS (MiB) | Mean busy cores, whole worker | Status |",
        "|---|---:|---:|---:|---|"]
    for engine, result in results.items():
        resource = result["resources"]
        duration = f"{result['sample_page_seconds']:.3f}" if "sample_page_seconds" in result else "Not completed"
        status = "Timeout" if resource["timed_out"] else ("Completed" if resource["exit_code"] == 0 else "Failed; see log")
        lines.append(f"| {engine} | {duration} | {resource['sampled_peak_tree_rss_mib']:.1f} | {resource['mean_busy_cpu_cores_whole_worker']:.2f} | {status} |")
    lines += ["", "RSS includes initialization and warmup. Summing process RSS can double-count shared pages. CPU is sampled across the whole worker, not isolated inference, and can miss short-lived child work.",
        "The thread setting is an engine configuration, not a hard OS core limit. Import, image and math libraries can use other threads.",
        "Paddle's effective detector/recognizer settings, stage timings and input shapes are in its JSON file when the worker completes."]
    (folder / "profile.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
