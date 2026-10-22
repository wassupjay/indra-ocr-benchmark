"""Fresh three-engine comparison on a seeded subset or all saved book pages."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import sys

from benchmark import ENGINES, load_dataset, save_json, summarize
from evaluate import sha256
from profile_ocr import monitor
from publication_report import audit_dataset


def select_pages(pages):
    return sorted(random.Random(20261018).sample(range(len(pages)), 12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-pages", action="store_true", help="Run all 114 source pages in a separate experiment")
    args = parser.parse_args()
    stem = "fullbook" if args.all_pages else "deployment"
    count_label = "114-page" if args.all_pages else "12-page"
    source = Path("eval/benchmarks/f960d85713a3-200dpi-all")
    original = load_dataset(source)
    selected = list(range(len(original["pages"]))) if args.all_pages else select_pages(original["pages"])
    folder = Path("eval/benchmarks") / ("fullbook-onnx-114" if args.all_pages else "deployment-onnx-12") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder.mkdir(parents=True)
    pages = []
    for number, index in enumerate(selected, 1):
        item = original["pages"][index]
        page = {**item, "page": number, "source_pdf_page": item["page"]}
        for kind, extension in (("image", "png"), ("reference", "txt")):
            name = f"page_{number:03}.{extension}"
            shutil.copyfile(source / item[kind], folder / name)
            page[kind] = name
        pages.append(page)
    manifest = {**original, "kind": "pdf_text_layer_subset", "pages": pages,
        "source_dataset_sha256": sha256(source / "dataset.json"), "seed": 20261018,
        "page_selection": "all source pages in original order; same PNG/reference bytes copied" if args.all_pages else "12 source page indices sampled with Python random.Random(20261018), sorted; same PNG/reference bytes copied, no score-dependent selection"}
    save_json(folder / "dataset.json", manifest)
    load_dataset(folder)
    status = {"phase": "running", "dataset": str(folder), "source_pdf_pages": [p["source_pdf_page"] for p in pages],
              "completed": False, "resources": {}}
    status_path = Path("publication") / f"{stem}-status.json"
    save_json(status_path, status)
    print(f"Deployment subset: {folder}; PDF pages {status['source_pdf_pages']}", flush=True)
    for engine in ENGINES:
        command = [sys.executable, "-u", "benchmark.py", "--engine", engine, "--dataset", str(folder),
            "--threads", "4", "--repeats", "3", "--paddle-backend", "onnx-ftz",
            "--paddle-model-dir", ".onnx-models/PP-OCRv6_medium"]
        print(f"Running {engine}", flush=True)
        resources = monitor(command, 10800 if args.all_pages else 1800, folder / f"{engine}.log")
        resources["scope"] = "entire engine worker, including imports, loading, excluded warmup, three passes, scoring and output writes; summed RSS can double-count shared memory"
        save_json(folder / f"{engine}-resources.json", resources)
        status["resources"][engine] = resources
        save_json(status_path, status)
        files = sorted((folder / engine).glob("*/results.json"))
        if resources["timed_out"] or resources["exit_code"] != 0 or not files:
            raise RuntimeError(f"{engine} did not finish; see {folder / (engine + '.log')}")
        result = json.loads(files[-1].read_text(encoding="utf-8"))
        if result["errors"]:
            raise RuntimeError(f"{engine} had failed pages; not promoting the comparison")
    audit = audit_dataset(folder)
    stats = json.loads((folder / "publication-stats.json").read_text(encoding="utf-8"))
    # Summaries contain aggregate metrics and hashes, not commercial book text.
    summary = {"completed": True, "dataset": str(folder), "dataset_sha256": sha256(folder / "dataset.json"),
        "source_pdf_pages": status["source_pdf_pages"], "statistics": stats,
        "resources": status["resources"], "audit": audit,
        "paddle_configuration": "PP-OCRv6 medium, converted same cached weights, ONNXRuntime CPU, four intra-op/one inter-op threads, session.set_denormal_as_zero=1",
        "limitation": f"{len(pages)} pages of one English book; PDF text-layer agreement, not independently verified accuracy; separate sequential engine runs, not interleaved"}
    save_json(Path("publication") / f"{stem}-comparison.json", summary)
    rows = [f"# Fresh {count_label} deployment comparison", "", summary["limitation"], "",
        "Three measured passes per engine after one first-page warmup. All engines receive the same saved 200-DPI RGB images. Four requested inference threads. Tesseract uses one CLI process per image; Python models remain loaded.", "",
        f"| Engine | WER | CER | Pages/min | Median {count_label} pass (s) | Peak tree RSS (MiB) |", "|---|---:|---:|---:|---:|---:|"]
    for engine, m in stats["models"].items():
        rows.append(f"| {engine} | {m['wer']:.3%} | {m['cer']:.3%} | {m['pages_per_minute']:.2f} | {m['median_pass_seconds']:.3f} | {status['resources'][engine]['sampled_peak_tree_rss_mib']:.1f} |")
    rows += ["", summary["paddle_configuration"], "", f"PDF pages: {status['source_pdf_pages']}. Selection seed: 20261018.",
        "", "RAM/CPU cover the entire worker, including initialization, warmup and scoring. Throughput excludes those stages. They are different measurement scopes. Summed process-tree RSS can double-count shared pages.",
        "", "Raw predictions were rescored and all input/output hashes checked. Numerical scores and normalized text hashes were saved for every pass. Original 114-page results remain separate."]
    (Path("publication") / f"{stem}-comparison.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    status.update(phase="completed", completed=True)
    save_json(status_path, status)
    print("\n".join(rows), flush=True)


if __name__ == "__main__":
    main()
