"""Validate saved synthetic predictions, including the separately completed image.

This produces quality-only results. It never reconstructs a complete-run speed.
"""

import json
from pathlib import Path

from benchmark import ENGINES, load_dataset, save_json
from evaluate import aggregate, compare, normalize, sha256


def validate(folder):
    manifest = load_dataset(folder)
    results = {}
    separate_predictions = []
    for engine in ENGINES:
        runs = sorted(p for p in (folder / engine).glob("*") if p.is_dir())
        if not runs:
            raise ValueError(f"No saved {engine} predictions.")
        run = runs[-1]
        pages = []
        for page in manifest["pages"]:
            prediction = run / f"page_{page['page']:03}.txt"
            if not prediction.exists() and engine == "paddleocr":
                matches = []
                for report in sorted((folder / "profiles").glob("*/paddleocr.json")):
                    data = json.loads(report.read_text(encoding="utf-8"))
                    if data.get("phase") != "completed" or data.get("sample_pdf_page") != page["page"]:
                        continue
                    settings = data.get("metadata", {}).get("settings", {})
                    if (settings.get("text_detection_model_name") != "PP-OCRv6_medium_det"
                        or settings.get("text_recognition_model_name") != "PP-OCRv6_medium_rec"
                        or settings.get("cpu_threads") != 4 or settings.get("enable_mkldnn") is not True
                        or data.get("metadata", {}).get("version") != "3.7.0"
                        or data.get("metadata", {}).get("paddlepaddle") != "3.2.2"
                        or "text_det_limit_side_len" in settings or "engine_config" in settings):
                        continue
                    if data.get("dataset_sha256", sha256(folder / "dataset.json")) != sha256(folder / "dataset.json"):
                        continue
                    if report.with_suffix(".txt").exists():
                        matches.append(report.with_suffix(".txt"))
                if matches:
                    prediction = matches[-1]
                    separate_predictions.append({"engine": engine, "page": page["page"], "prediction_file": str(prediction)})
            if not prediction.exists():
                raise ValueError(f"Missing {engine} prediction for control image {page['page']}")
            reference = (folder / page["reference"]).read_text(encoding="utf-8")
            text = prediction.read_text(encoding="utf-8")
            pages.append({"page": page["page"], "variant": page["variant"],
                          "prediction_file": str(prediction), "prediction_sha256": sha256(prediction),
                          "score": compare(normalize(reference, []), normalize(text, []))})
        results[engine] = {"overall": aggregate(pages, "score"), "items": pages}
    output = {"dataset_sha256": sha256(folder / "dataset.json"), "models": results,
        "scope": "quality-only validation on three synthetic variants of one original passage; not independent documents",
        "separately_completed_predictions": separate_predictions,
        "paddle_provenance": ("Saved PaddleOCR predictions from the interrupted full control attempt were supplemented with separately completed single-image diagnostics using the same model names and default settings. See the per-image provenance. No combined throughput is claimed."
            if separate_predictions else "All predictions came from saved engine runs. This quality-only validation does not recompute throughput.")}
    save_json(folder / "quality-validation.json", output)
    lines = ["# Known-text control: completed quality check", "",
        "Reference text is known by construction. All models saw the same three images.", "",
        "| Engine | Reference words (repeated across variants) | Word edits | Character edits | WER | CER |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, result in results.items():
        s = result["overall"]
        lines.append(f"| {name} | {s['reference_words']} | {s['word_errors']} | {s['character_errors']} | {s['wer']:.3%} | {s['cer']:.3%} |")
    lines += ["", output["paddle_provenance"], "",
        "This is a small clean-print sanity check. Three font/blur variants repeat the same 124-word passage. Zero edits here do not imply general OCR accuracy or equivalence on real documents.",
        "Raw predictions and their hashes are listed in quality-validation.json. The public annotated-document evaluation remains pending download access."]
    (folder / "quality-validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    folder = Path("eval/benchmarks/synthetic-control-v1")
    validate(folder)
    print(f"Saved {folder / 'quality-validation.md'}")
