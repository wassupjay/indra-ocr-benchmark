"""Publication statistics from saved results. Does not run OCR."""

import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import random
import statistics


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lo = int(position)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (position - lo)


def rate_samples(errors, denominators, samples):
    rates = []
    for indices in samples:
        denominator = sum(denominators[i] for i in indices)
        if denominator:
            rates.append(sum(errors[i] for i in indices) / denominator)
    return rates


def interval(values):
    return [percentile(values, .025), percentile(values, .975)]


def build_statistics(folder, replicates=5000):
    source = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    results = source["results"]
    if not results:
        raise ValueError("No completed runs to summarize.")
    page_ids = [p["page"] for p in results[0]["pages"]]
    denominators = {metric: [p["score"][key] for p in results[0]["pages"]]
                    for metric, key in (("wer", "reference_words"), ("cer", "reference_characters"))}
    rng = random.Random(20261018)
    samples = [rng.choices(range(len(page_ids)), k=len(page_ids)) for _ in range(replicates)]
    output = {"reference_status": source["dataset"]["reference_status"],
              "dataset_kind": source["dataset"].get("kind", "pdf_text_layer"),
              "unit": source["dataset"].get("unit", "pages"),
              "uncertainty_method": f"paired percentile item bootstrap, {replicates} resamples, seed 20261018; descriptive within-sample uncertainty only; source pages or crops are not necessarily independent documents",
              "models": {}, "paired_comparisons": [], "missing_engines": source["missing_engines"]}
    distributions = {}
    csv_rows = []
    for result in results:
        name = result["engine"]
        pages = result["pages"]
        if [p["page"] for p in pages] != page_ids:
            raise ValueError("Runs do not have identical ordered page IDs.")
        for metric, key in (("wer", "reference_words"), ("cer", "reference_characters")):
            if [p["score"][key] for p in pages] != denominators[metric]:
                raise ValueError("Runs use different reference denominators.")
        distributions[name] = {}
        for metric, key in (("wer", "word_errors"), ("cer", "character_errors")):
            distributions[name][metric] = rate_samples([p["score"][key] for p in pages], denominators[metric], samples)
        latencies = [p["seconds"] for p in pages]
        counts = {key: sum(p["score"]["counts"][key] for p in pages) for key in ("replace", "delete", "insert")}
        if sum(counts.values()) != result["overall"]["word_errors"]:
            raise ValueError("Word edit breakdown does not match total edits.")
        model = {
            "wer": result["overall"]["wer"], "cer": result["overall"]["cer"],
            "wer_page_bootstrap_95": interval(distributions[name]["wer"]),
            "cer_page_bootstrap_95": interval(distributions[name]["cer"]),
            "word_edits": result["overall"]["word_errors"], "word_edit_breakdown": counts,
            "exact_text_pages": sum(p["score"]["character_errors"] == 0 for p in pages),
            "pages": len(pages), "failures_all_passes": len(result["errors"]),
            "latency_first_pass_seconds": {"mean": statistics.mean(latencies), "p50": percentile(latencies, .5),
                "p95": percentile(latencies, .95), "max": max(latencies), "min": min(latencies)},
            "pass_seconds": result["pass_seconds"],
            "pass_time_range_seconds": [min(result["pass_seconds"]), max(result["pass_seconds"])],
            "pass_time_cv_percent": statistics.stdev(result["pass_seconds"]) / statistics.mean(result["pass_seconds"]) * 100
                if len(result["pass_seconds"]) > 1 else None,
            "median_pass_seconds": result["median_pass_seconds"], "pages_per_minute": result["pages_per_minute"],
            "initialization_seconds": result["initialization_seconds"], "metadata": result["metadata"],
            "hardware": result["hardware"],
            "resource_measurements_full_run": (
                json.loads((folder / f"{name}-resources.json").read_text(encoding="utf-8"))
                if (folder / f"{name}-resources.json").is_file() else
                "not recorded in this result; consult separate resource logs if present; historical resources cannot be reconstructed"),
            "worst_wer_pages": [{"page": p["page"], "wer": p["score"]["wer"]}
                for p in sorted(pages, key=lambda p: p["score"]["wer"] or 0, reverse=True)[:5]],
        }
        output["models"][name] = model
        for page in pages:
            csv_rows.append({"engine": name, "page": page["page"], "seconds_first_pass": page["seconds"],
                **{k: page["score"][k] for k in ("wer", "cer", "word_errors", "character_errors", "reference_words", "reference_characters")},
                **page["score"]["counts"]})
    for left, right in combinations(results, 2):
        a, b = left["engine"], right["engine"]
        comparison = {"left": a, "right": b}
        for metric in ("wer", "cer"):
            differences = [(x - y) * 100 for x, y in zip(distributions[a][metric], distributions[b][metric])]
            comparison[f"{metric}_difference_percentage_points"] = (left["overall"][metric] - right["overall"][metric]) * 100
            comparison[f"{metric}_difference_95"] = interval(differences)
        comparison["page_word_edit_wins_left_ties_right"] = [
            sum(x["score"]["word_errors"] < y["score"]["word_errors"] for x, y in zip(left["pages"], right["pages"])),
            sum(x["score"]["word_errors"] == y["score"]["word_errors"] for x, y in zip(left["pages"], right["pages"])),
            sum(x["score"]["word_errors"] > y["score"]["word_errors"] for x, y in zip(left["pages"], right["pages"])),
        ]
        output["paired_comparisons"].append(comparison)
    (folder / "publication-stats.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    with (folder / "page-stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    write_markdown(folder, output)
    return output


def write_markdown(folder, output):
    lines = ["# OCR comparison: publication statistics", "",
        "Reference status: " + output["reference_status"] + ".", "",
        "## Recognition agreement", "",
        "| Engine | WER | WER page-bootstrap 95% interval | CER | Word edits | Substitutions | Deletions | Insertions | Exact-text pages |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, model in output["models"].items():
        lo, hi = model["wer_page_bootstrap_95"]
        counts = model["word_edit_breakdown"]
        lines.append(f"| {name} | {model['wer']:.3%} | {lo:.3%}–{hi:.3%} | {model['cer']:.3%} | {model['word_edits']} | {counts['replace']} | {counts['delete']} | {counts['insert']} | {model['exact_text_pages']}/{model['pages']} |")
    lines += ["", "## Recorded application throughput", "",
        "| Engine | Pages/min | Median full pass (s) | Pass range (s) | First-pass page p50 (s) | First-pass page p95 (s) | Initialization (s) | Failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, model in output["models"].items():
        lo, hi = model["pass_time_range_seconds"]
        latency = model["latency_first_pass_seconds"]
        lines.append(f"| {name} | {model['pages_per_minute']:.2f} | {model['median_pass_seconds']:.2f} | {lo:.2f}–{hi:.2f} | {latency['p50']:.3f} | {latency['p95']:.3f} | {model['initialization_seconds']:.2f} | {model['failures_all_passes']} |")
    lines += ["", "Page latency percentiles in this table use the first pass only. Newer results also preserve later per-page timings in pass_details; historical runs may contain pass totals only.",
        "Three pass totals describe repeatability, not a reliable speed confidence interval. Initialization excludes the first warmup call.",
        "Resource scope is recorded separately from inference-only timing. Resource profiles for a different run or a small sample must not be relabeled as full-run memory measurements. Historical resources cannot be reconstructed when no monitoring record exists.",
        "", "## Paired accuracy differences", "",
        "Negative differences favor the left engine. Intervals are descriptive within-sample bootstrap estimates, not evidence of performance across other independent documents.", "",
        "| Left vs right | WER difference (percentage points) | Paired 95% interval | Page wins / ties / losses |",
        "|---|---:|---:|---:|"]
    for pair in output["paired_comparisons"]:
        lo, hi = pair["wer_difference_95"]
        wins = " / ".join(map(str, pair["page_word_edit_wins_left_ties_right"]))
        lines.append(f"| {pair['left']} vs {pair['right']} | {pair['wer_difference_percentage_points']:.4f} | {lo:.4f}–{hi:.4f} | {wins} |")
    lines += ["", "## What can be claimed", "",
        f"These are measured results on the same {next(iter(output['models'].values()))['pages']} {output['unit']}. Reference: {output['reference_status']}.",
        "Do not call 100−WER a probability of correct recognition. Edit rates include insertions and can exceed 100%.",
        "Do not claim a general winner from these results. Diverse independent documents and validated runtime configurations remain required for broader claims.",
        "Tesseract CLI starts once per page; Python engines remain resident. Reported throughput includes that integration difference.",
        "Model versions, settings and hardware details are in publication-stats.json. Reproduction commands and limitations are in BENCHMARK.md."]
    (folder / "publication-stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path, default=Path("eval/benchmarks/f960d85713a3-200dpi-all"))
    args = parser.parse_args()
    result = build_statistics(args.folder)
    print(f"Saved {args.folder / 'publication-stats.md'}")
    for pair in result["paired_comparisons"]:
        print(pair["left"], "vs", pair["right"], "WER difference 95% interval (percentage points):", pair["wer_difference_95"])
