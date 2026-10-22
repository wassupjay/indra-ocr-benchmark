"""Synthetic clean-print control with exact text known by construction.

This is not a public scanned-document benchmark and does not replace one.
"""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont, ImageFilter

from benchmark import ENGINES, run_engine, save_json
from evaluate import sha256
from profile_ocr import monitor


TEXT = """A small town library
The library opens at nine in the morning and closes at six in the evening. On Saturday, the reading room stays open until four. Visitors can borrow three books for fourteen days. Please return books to the front desk.
Monthly account summary
The opening balance was 1,250.00 dollars. A deposit of 375.50 dollars arrived on March 12, 2026. The next payment was 84.25 dollars. The closing balance was 1,541.25 dollars. Reference number: AB-2048.
Notes for the next meeting
Bring the blue folder, a pencil, and the revised schedule. Compare the old and new totals before signing the form. The room is on the second floor, beside the north stairs. Questions are welcome: clear instructions help everyone finish the work correctly."""


def wrap(draw, text, font, width):
    lines = []
    for paragraph in text.splitlines():
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if draw.textlength(candidate, font=font) > width:
                if not line:
                    raise ValueError("A word is wider than the rendering area.")
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.extend([line, ""])
    return lines


def prepare_control():
    folder = Path("eval/benchmarks/synthetic-control-v1")
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "dataset.json").exists():
        manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
        for page in manifest["pages"]:
            for key in ("image", "reference"):
                if sha256(folder / page[key]) != page[f"{key}_sha256"]:
                    raise ValueError("Synthetic control changed after generation.")
        return folder, manifest
    specs = [("serif", "times.ttf", 28, 0), ("sans", "arial.ttf", 28, 0),
             ("small_monospace_blur", "consola.ttf", 24, .5)]
    pages = []
    for number, (label, filename, size, blur) in enumerate(specs, 1):
        font_path = Path("C:/Windows/Fonts") / filename
        font = ImageFont.truetype(str(font_path), size)
        image = Image.new("RGB", (900, 1100), "white")
        draw = ImageDraw.Draw(image)
        lines = wrap(draw, TEXT, font, 800)
        step = size + 9
        if 50 + len(lines) * step > image.height - 30:
            raise ValueError("Text would be clipped. Increase the control image height.")
        for index, line in enumerate(lines):
            draw.text((50, 50 + index * step), line, fill="black", font=font)
        if blur:
            image = image.filter(ImageFilter.GaussianBlur(blur))
        image_path, reference_path = folder / f"page_{number:03}.png", folder / f"page_{number:03}.txt"
        image.save(image_path, dpi=(200, 200))
        reference_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        pages.append({"page": number, "image": image_path.name, "reference": reference_path.name,
            "image_sha256": sha256(image_path), "reference_sha256": sha256(reference_path),
            "variant": label, "font": filename, "font_sha256": sha256(font_path),
            "font_size_pixels": size, "gaussian_blur_radius": blur})
    manifest = {"kind": "synthetic_control", "reference_status": "exact original text used to render these images; known by construction",
        "limitation": "three rendering variants of the same original English passage, not independent real documents",
        "pillow_version": version("Pillow"), "dpi": 200, "pages": pages}
    save_json(folder / "dataset.json", manifest)
    (folder / "source.txt").write_text(TEXT + "\n", encoding="utf-8")
    return folder, manifest


def summarize(folder, manifest):
    lines = ["# Synthetic control: exact reference text", "",
        "Three rendering variants of one original English passage. Ground truth is the exact text used to draw each image.",
        "This tests clean-print recognition under controlled conditions, not accuracy on real scans or diverse documents.", "",
        "| Engine | WER | CER | Word edits | Pages/min | Measured passes |",
        "|---|---:|---:|---:|---:|---:|"]
    records = {}
    for engine in ENGINES:
        files = sorted((folder / engine).glob("*/results.json"))
        if not files:
            lines.append(f"| {engine} | Incomplete | — | — | — | — |")
            continue
        result = json.loads(files[-1].read_text(encoding="utf-8"))
        if result["dataset_sha256"] != sha256(folder / "dataset.json"):
            raise ValueError("Control result uses a different manifest.")
        records[engine] = result
        score = result["overall"]
        lines.append(f"| {engine} | {score['wer']:.3%} | {score['cer']:.3%} | {score['word_errors']} | {result['pages_per_minute']:.2f} | {len(result['pass_seconds'])} |")
    lines += ["", "All engines see identical images. Default engine settings match the book experiment. Initialization and one first-page warmup are excluded from throughput.",
        "Only one pass is measured here. Do not treat three near-identical synthetic pages as evidence of a general winner or compute a population confidence interval from them.",
        "The manuscript was authored for this test; no book text is used. Font files are not redistributed. Font hashes and rendering settings are in dataset.json."]
    if (folder / "quality-validation.md").exists():
        lines += ["", "A separate [completed quality check](quality-validation.md) includes predictions completed outside the timed control pass. That does not turn an interrupted pass into a complete speed measurement."]
    save_json(folder / "control-results.json", {"dataset": manifest, "results": records})
    (folder / "control-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=(*ENGINES, "all", "summary"), default="all")
    parser.add_argument("--timeout", type=float, default=480)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    folder, manifest = prepare_control()
    if args.worker:
        run_engine(args.engine, folder, manifest, 4, 1)
        return
    if args.engine != "summary":
        for engine in ENGINES if args.engine == "all" else [args.engine]:
            diagnostic = folder / (engine + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
            print(f"Running synthetic control: {engine}", flush=True)
            resource = monitor([sys.executable, str(Path(__file__).resolve()), "--worker", "--engine", engine],
                args.timeout, diagnostic.with_suffix(".log"))
            save_json(diagnostic.with_suffix(".json"), resource)
            print(f"{engine}: exit={resource['exit_code']}, timeout={resource['timed_out']}", flush=True)
    summarize(folder, manifest)


if __name__ == "__main__":
    main()
