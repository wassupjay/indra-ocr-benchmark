"""Audit saved predictions and build the shareable publication tables."""

import argparse
import csv
from datetime import datetime, timezone
import html
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from benchmark import ENGINES, dataset_engines, load_dataset, save_json, summarize
from evaluate import aggregate, compare, normalize, sha256
from validate_control import validate


BOOK = Path("eval/benchmarks/f960d85713a3-200dpi-all")
CONTROL = Path("eval/benchmarks/synthetic-control-v1")


def audit_dataset(folder):
    manifest = load_dataset(folder)
    evidence = {}
    for engine in dataset_engines(manifest):
        files = sorted((folder / engine).glob("*/results.json"))
        if not files:
            raise ValueError(f"Missing completed {engine} run in {folder}")
        source = files[-1]
        result = json.loads(source.read_text(encoding="utf-8"))
        if result["dataset_sha256"] != sha256(folder / "dataset.json"):
            raise ValueError(f"Changed dataset for {engine}")
        if [p["page"] for p in result["pages"]] != [p["page"] for p in manifest["pages"]]:
            raise ValueError(f"Incomplete page list for {engine}")
        pages, hashes, normalized_hashes = [], {}, {}
        for item, saved in zip(manifest["pages"], result["pages"]):
            prediction = source.parent / f"page_{item['page']:03}.txt"
            reference = (folder / item["reference"]).read_text(encoding="utf-8")
            text = prediction.read_text(encoding="utf-8")
            score = compare(normalize(reference, []), normalize(text, []))
            if score != saved["score"]:
                raise ValueError(f"Saved prediction/score mismatch: {engine}, page {item['page']}")
            pages.append({"score": score})
            hashes[prediction.name] = sha256(prediction)
            normalized_hashes[item["page"]] = hashlib.sha256(normalize(text, []).encode("utf-8")).hexdigest()
        overall = aggregate(pages, "score")
        if overall != result["overall"]:
            raise ValueError(f"Aggregate mismatch for {engine}")
        passes = result["pass_seconds"]
        if len(passes) < 3 or not all(math.isfinite(s) and s > 0 for s in passes):
            raise ValueError(f"Need three valid measured passes for {engine}")
        import statistics
        median = statistics.median(passes)
        if not math.isclose(sum(p["seconds"] for p in result["pages"]), passes[0], rel_tol=1e-9):
            raise ValueError(f"First-pass page timings do not add up for {engine}")
        if not math.isclose(result["median_pass_seconds"], median, rel_tol=1e-12):
            raise ValueError(f"Incorrect median duration for {engine}")
        if not math.isclose(result["pages_per_minute"], len(pages) * 60 / median, rel_tol=1e-12):
            raise ValueError(f"Incorrect throughput for {engine}")
        stable_outputs = None
        if "pass_details" in result:
            details = result["pass_details"]
            if len(details) != len(passes):
                raise ValueError("Per-pass measurements are incomplete")
            stable_outputs = True
            for number, trial in enumerate(details):
                if [p["page"] for p in trial["pages"]] != [p["page"] for p in manifest["pages"]]:
                    raise ValueError("Per-pass page list changed")
                if aggregate(trial["pages"], "score") != trial["overall"]:
                    raise ValueError("Per-pass aggregate does not match page scores")
                if not math.isclose(sum(p["seconds"] for p in trial["pages"]), passes[number], rel_tol=1e-9):
                    raise ValueError("Per-pass page timings do not add up")
                matches = all(p["normalized_text_sha256"] == normalized_hashes[p["page"]] for p in trial["pages"])
                if number == 0 and not matches:
                    raise ValueError("First-pass hashes do not match saved predictions")
                for page, saved in zip(trial["pages"], result["pages"]):
                    expected = {k: v for k, v in saved["score"].items() if k != "changes"}
                    if page["normalized_text_sha256"] == normalized_hashes[page["page"]] and page["score"] != expected:
                        raise ValueError("Identical per-pass output has a different score")
                stable_outputs = stable_outputs and matches
        evidence[engine] = {"result_file": source.as_posix(), "result_sha256": sha256(source),
            "prediction_sha256": hashes, "pages_recomputed": len(pages),
            "overall": overall, "execution_failures_all_passes": len(result["errors"]),
            "all_pass_outputs_match_saved_first_pass": stable_outputs}
    summarize(folder, manifest)
    return {"dataset_sha256": sha256(folder / "dataset.json"),
            "reference_status": manifest["reference_status"], "engines": evidence}


def compare_saved_outputs(current, earlier):
    comparison = {}
    for engine, evidence in current["engines"].items():
        previous = earlier["engines"][engine]
        names = sorted(evidence["prediction_sha256"])
        if names != sorted(previous["prediction_sha256"]):
            raise ValueError("Saved output comparisons need identical page filenames")
        current_dir = Path(evidence["result_file"]).parent
        previous_dir = Path(previous["result_file"]).parent
        changed = [name for name in names if normalize((current_dir / name).read_text(encoding="utf-8"), [])
                   != normalize((previous_dir / name).read_text(encoding="utf-8"), [])]
        comparison[engine] = {"pages_compared": len(names), "all_normalized_outputs_match": not changed,
                              "changed_prediction_files": changed}
    return comparison


def table_image(path, title, headers, rows, notes):
    """A table image for Medium, not a plot or an inferred performance chart."""
    font_path = "C:/Windows/Fonts/arial.ttf"
    font = ImageFont.truetype(font_path, 27)
    small = ImageFont.truetype(font_path, 23)
    heading = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 38)
    width, margin, row_height = 1700, 55, 76
    height = 150 + row_height * (len(rows) + 1) + 48 * len(notes) + 40
    image = Image.new("RGB", (width, height), "#fafaf7")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 40), title, fill="#16252c", font=heading)
    cell_width = (width - 2 * margin) / len(headers)
    for i, row in enumerate([headers, *rows]):
        y = 122 + i * row_height
        draw.rectangle((margin, y, width - margin, y + row_height), fill="#e5eef1" if i == 0 else ("#ffffff" if i % 2 else "#f1f4f4"))
        for j, value in enumerate(row):
            if draw.textlength(str(value), font=font) > cell_width - 20:
                raise ValueError(f"Table cell would overflow: {value}")
            draw.text((margin + j * cell_width + 12, y + 23), str(value), fill="#16252c", font=font)
    y = 145 + row_height * (len(rows) + 1)
    for note in notes:
        if draw.textlength(note, font=small) > width - 2 * margin:
            raise ValueError("Table note would overflow")
        draw.text((margin, y), note, fill="#4a5155", font=small)
        y += 48
    image.save(path)


def fullbook_article(run, audit, onnx, public_dataset=None):
    """Lead with the completed experiment, keeping historical results separate."""
    output = Path("publication")
    models = run["statistics"]["models"]
    rows = [[name + (" (ONNX)" if name == "paddleocr" else ""), f"{m['wer']:.3%}",
             f"{m['cer']:.3%}", f"{m['pages_per_minute']:.2f}", str(m["failures_all_passes"])]
            for name, m in models.items()]
    table_image(output / "fullbook-results.png", "114-page Windows CPU comparison: current run",
        ["Engine", "WER", "CER", "Pages/min", "Failures"], rows,
        ["Reference: unverified embedded PDF text. These are text-agreement scores.",
         "Same 114 RGB page images, 200 DPI. Three measured passes; one warmup excluded.",
         "Paddle: PP-OCRv6 medium, ONNX Runtime CPU, denormal-as-zero enabled.",
         "Serial image-to-text throughput; Tesseract CLI startup included."])
    source = Path(run["dataset"])
    (output / "fullbook-page-stats.csv").write_bytes((source / "page-stats.csv").read_bytes())
    lines = ["# Indra, Tesseract and PaddleOCR: a reproducible Windows CPU case study", "",
        "Three OCR engines processed the same 114 page images from one English book. This report compares their text agreement, application throughput, timing variation and sampled resource use. It includes a complete fresh run with PaddleOCR's ONNX CPU backend, separately from the earlier native-runtime measurements.", "",
        "**Scope:** the reference is the PDF's unverified embedded text, so WER/CER below measure agreement rather than independently verified accuracy. This study does not establish a general OCR winner.", "",
        "## Current full-book results", "", "![Current 114-page results](fullbook-results.png)", "",
        "| Engine | WER | CER | Pages/min | Failures across three passes |", "|---|---:|---:|---:|---:|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    lines += ["", "WER and CER are pooled edit counts divided by reference word/character counts, with case and punctuation significant. Unicode NFC, whitespace collapse and Markdown heading-prefix removal are applied consistently. Headers and footers count. Rates are not averaged across page percentages; 100 minus WER is not a calibrated probability of correct recognition.", "",
        "## Speed, repeatability and resources", "",
        "| Engine | Passes (s) | Pass-time CV | Page p50 / p95 (s) | Peak tree RSS (MiB) | Mean busy CPU cores |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, m in models.items():
        resources = run["resources"][name]
        latency = m["latency_first_pass_seconds"]
        lines.append(f"| {name} | {' / '.join(f'{s:.3f}' for s in m['pass_seconds'])} | {m['pass_time_cv_percent']:.2f}% | {latency['p50']:.3f} / {latency['p95']:.3f} | {resources['sampled_peak_tree_rss_mib']:.1f} | {resources['mean_busy_cpu_cores_whole_worker']:.2f} |")
    lines += ["", "Throughput uses the median of three complete-pass image-to-text durations. It includes image decoding and Tesseract's per-page CLI startup; the Python engines remain loaded. It excludes rendering, model loading, the first-image warmup, scoring and writes. Per-page percentiles above use the first pass. Each engine ran sequentially; engines were not interleaved or tested under an isolated operating system.", "",
        "Resources were sampled every 100 ms across the entire engine worker, including imports, model loading, warmup, all three passes, scoring and writes. Mean busy cores is sampled CPU seconds divided by that whole-worker wall time. Summed process-tree RSS can double-count shared memory; sampling can miss brief peaks and short-lived child CPU work. Resource and throughput scopes differ.", "",
        "An earlier 12-page check showed substantial unexplained Indra timing variation (38.77% pass-time CV). Historical and current speeds also differ. The pass durations above expose repeatability in this run; they do not explain cross-session drift or establish a universally stable speed ranking.", "",
        "## Recognition differences and uncertainty", "",
        "| Comparison | WER difference (percentage points) | Paired 95% interval |", "|---|---:|---:|"]
    for pair in run["statistics"]["paired_comparisons"]:
        lo, hi = pair["wer_difference_95"]
        lines.append(f"| {pair['left']} minus {pair['right']} | {pair['wer_difference_percentage_points']:.4f} | {lo:.4f} to {hi:.4f} |")
    lines += ["", "Intervals use 5,000 paired page-bootstrap resamples with seed 20261018. Negative differences favor the left engine. These are descriptive intervals within this one book; pages are not independent books, and these intervals do not justify claims across document types.", "",
        "## What ran", "",
        "Windows build 26200, Intel Core i7-12700K, 20 logical CPUs and approximately 95.75 GiB RAM. Four inference threads were requested per engine; this does not cap every library or operating-system thread. Indra 0.3.2 ran in forced OCR mode with furniture enabled and plain page.text output. Tesseract 5.5.3 used English, OEM 1 and PSM 3. PaddleOCR 3.7.0 / PaddleX 3.7.2 used PP-OCRv6 medium detection and recognition, with document orientation, unwarping and text-line orientation disabled.", "",
        "Paddle used ONNX Runtime 1.31.0, CPUExecutionProvider, four intra-op and one inter-op threads, and session.set_denormal_as_zero=1. Its original cached weights were converted to FP32 ONNX using Paddle2ONNX 2.1.0 with the exact compatible Paddle 3.0.0.dev20250426 converter runtime. Both models passed full ONNX checks. There was no quantization or FP16 conversion. The installed inference PaddlePaddle 3.2.2 environment was not replaced by the isolated converter.", "",
        "The [ONNX Runtime setting](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h) changes handling of very small floating-point values and can affect numerical outputs. Every measured pass was scored and its normalized output hashes recorded. Neither structural validation nor agreement on tested images proves numerical equivalence for arbitrary inputs. Both OpenCV distributions are installed; the exporter records the actual imported cv2 runtime separately.", "",
        "## What changed from the earlier measurements", "",
        "The original native Paddle run achieved 0.84 pages/min on these book images. It did not record weight hashes or full-run resources. That historical result remains in book-results.csv and the earlier statistics; it is not combined with current timings into one ranking."]
    if onnx:
        lines += ["", f"On three known-text synthetic controls, separate repeated runs measured a {onnx['observed_speed_ratio']:.2f}x plain-Paddle-to-ONNX speed ratio with zero text edits in either configuration. These are three renderings of the same 124-word passage, not independent documents. A default-ONNX diagnostic was incomplete; only its first measured image was timed. The supported denormal setting substantially improved the tested configuration, but the exact kernel-level cause and maximum possible speed remain unproven."]
    quality_path = output / "fullbook-quality-check.json"
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        lines += ["", "Current versus historical first-pass normalized outputs: " + "; ".join(
            f"{name}: {m['pages_compared'] - len(m['changed_prediction_files'])}/{m['pages_compared']} pages match"
            for name, m in quality.items()) + ". The changed Paddle page is listed in fullbook-quality-check.json. Aggregate edit counts can remain equal despite different text. Historical Paddle weight identity was not recorded, so this compares saved outputs without attributing the difference solely to ONNX or the denormal setting."]
    lines += ["", "## Evidence and reproduction", "",
        f"Dataset manifest SHA256: `{audit['dataset_sha256']}`. All {sum(e['pages_recomputed'] for e in audit['engines'].values())} saved first-pass predictions were rescored from text; input/output hashes, aggregate scores, pass sums, medians and throughput were checked. Numerical scores and normalized output hashes are saved for all nine engine passes. The evidence audit reports whether later outputs match the first pass.", "",
        "ONNX_REPRO.md gives the tested setup and commands. run_deployment_comparison.py --all-pages creates a fresh, separate 114-page experiment. fullbook-page-stats.csv contains page-level scores and latency. fullbook-comparison.json contains configurations, statistics and measured resources. The release ZIP contains code, permitted aggregate artifacts, original synthetic controls, file checksums and a current environment snapshot.", "",
        "The commercial book, its page images, embedded text and raw OCR predictions are excluded from the shareable archive. Reproducing this experiment requires a legally obtained PDF matching the recorded hash. A public hosting URL and Medium posting are still separate publication steps.", "",
        "## Limits and next evidence", "",
        "The book text layer has not been independently verified. A diverse dataset with independently provided annotations is still needed for broader accuracy claims. Clean synthetic controls cannot supply that evidence. Training overlap, language/domain coverage, scan degradation and full-page reading order are not established by this experiment."]
    if public_dataset:
        public = json.loads((public_dataset / "publication-stats.json").read_text(encoding="utf-8"))
        lines += ["", "## Annotated text-region follow-up", "", "Custom OmniDocBench recognition crops with supplied gold boundaries; not an official full-page score. Possible annotation errors and training overlap remain. Region throughput is separate from full-page throughput.", "", "| Engine | WER | CER | Regions/min |", "|---|---:|---:|---:|"]
        lines += [f"| {name} | {m['wer']:.3%} | {m['cer']:.3%} | {m['pages_per_minute']:.2f} |" for name, m in public["models"].items()]
    else:
        lines += ["", "The prepared OmniDocBench annotated-region follow-up could not download data in this execution environment (WinError 10061). No public annotated accuracy scores are claimed. finish_publication.py resumes this separate follow-up when normal network access is available."]
    lines += ["", "The measured results support a reproducible Windows case study and a tested Paddle execution configuration. They do not establish maximum speed, universal numerical equivalence or an industry-wide accuracy ranking."]
    article = "\n".join(lines) + "\n"
    (output / "fullbook-comparison.md").write_text(article, encoding="utf-8")
    overview = '<h2>Current full-book comparison</h2><p><a href="fullbook-comparison.md">114-page results, repeatability and resources</a></p><img src="fullbook-results.png" alt="Current 114-page CPU OCR comparison">'
    return article, overview


def build(public_dataset=None):
    output = Path("publication")
    output.mkdir(exist_ok=True)
    book_audit = audit_dataset(BOOK)
    control = validate(CONTROL)
    audit = {"checked_utc": datetime.now(timezone.utc).isoformat(), "book": book_audit,
             "control_sha256": sha256(CONTROL / "quality-validation.json"),
             "scope": "recomputed saved first-pass predictions and validated three pass totals; no inference rerun",
             "public_annotated_results": False,
             "paddle_maximum_possible_throughput_established": False,
             "historical_full_book_resources_recorded": False}
    if public_dataset:
        if load_dataset(public_dataset).get("kind") != "public_annotated_text_regions":
            raise ValueError("Public follow-up must use an annotated-region dataset")
        audit["public"] = audit_dataset(public_dataset)
        audit["public_annotated_results"] = True
    verification_file = output / "paddle-verification.json"
    verification = json.loads(verification_file.read_text(encoding="utf-8")) if verification_file.is_file() else None
    if verification:
        audit["paddle_control_configuration_verified"] = verification["configuration_verified_on_control"]
        audit["paddle_verification_sha256"] = sha256(verification_file)
    onnx_file = output / "onnx-comparison.json"
    onnx = json.loads(onnx_file.read_text(encoding="utf-8")) if onnx_file.is_file() else None
    deployment_file = output / "deployment-comparison.json"
    deployment = json.loads(deployment_file.read_text(encoding="utf-8")) if deployment_file.is_file() else None
    if onnx:
        audit["onnx_control_comparison_sha256"] = sha256(onnx_file)
        audit["onnx_control_outputs_match_baseline"] = onnx["normalized_outputs_match_baseline"]
    if deployment:
        audit["fresh_deployment"] = audit_dataset(Path(deployment["dataset"]))
        audit["fresh_deployment_comparison_sha256"] = sha256(deployment_file)
    fullbook_file = output / "fullbook-comparison.json"
    fullbook = json.loads(fullbook_file.read_text(encoding="utf-8")) if fullbook_file.is_file() else None
    if fullbook:
        if not fullbook.get("completed") or len(load_dataset(Path(fullbook["dataset"]))["pages"]) != 114:
            raise ValueError("Full-book comparison must contain 114 completed pages")
        audit["fresh_fullbook"] = audit_dataset(Path(fullbook["dataset"]))
        audit["fresh_fullbook_comparison_sha256"] = sha256(fullbook_file)
        quality = compare_saved_outputs(audit["fresh_fullbook"], book_audit)
        save_json(output / "fullbook-quality-check.json", quality)
        audit["fullbook_historical_output_comparison"] = quality
    cpu_path = output / "cpu-four-comparison.json"
    cpu = json.loads(cpu_path.read_text(encoding="utf-8")) if cpu_path.is_file() else None
    if cpu:
        cpu_folder = Path(cpu["dataset"])
        if not cpu.get("completed") or not cpu_folder.resolve().is_relative_to(Path("eval/benchmarks/cpu-four-114").resolve()):
            raise ValueError("Four-engine publication needs a completed run in its benchmark directory")
        audit["four_engine_cpu"] = audit_dataset(cpu_folder)
        if set(audit["four_engine_cpu"]["engines"]) != {"indra", "tesseract", "easyocr", "paddleocr_tiny"}:
            raise ValueError("Four-engine publication is missing a requested engine")
        audit["four_engine_comparison_sha256"] = sha256(cpu_path)
    transcription_file = output / "transcription-comparison.json"
    if transcription_file.is_file():
        transcription = json.loads(transcription_file.read_text(encoding="utf-8"))
        if (sha256(Path("Rich_Dad_Poor_Dad_transcription.md")) != transcription["source_sha256"]
                or sha256(cpu_path) != transcription["source_comparison_sha256"]):
            raise ValueError("Candidate transcription analysis is stale; rerun evaluate_transcription.py")
        audit["candidate_transcription"] = {"report_sha256": sha256(transcription_file),
            "reference_status": transcription["reference_status"], "source_sha256": transcription["source_sha256"]}
    save_json(output / "evidence-audit.json", audit)
    stats = json.loads((BOOK / "publication-stats.json").read_text(encoding="utf-8"))
    rows = []
    for name, model in stats["models"].items():
        rows.append([name, f"{model['wer']:.3%}", f"{model['cer']:.3%}",
                     f"{model['pages_per_minute']:.2f}", str(model["failures_all_passes"])])
    table_image(output / "book-results.png", "114-page Windows CPU case study",
                ["Engine", "WER", "CER", "Pages/min", "Failures"], rows,
                ["Reference: unverified PDF text layer. WER/CER measure agreement, not verified accuracy.",
                 "Speed: median of 3 measured passes; Tesseract process startup included.",
                 "PaddleOCR: historical native runtime; later ONNX measurements are reported separately."])
    with (output / "book-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["engine", "WER", "CER", "recorded_pages_per_min", "failures_all_passes"])
        writer.writerows(rows)
    article = Path("MEDIUM_DRAFT.md").read_text(encoding="utf-8")
    article = article.replace("## Text agreement results", "![Measured book results](book-results.png)\n\n## Text agreement results")
    if deployment:
        article = article.replace("## Recorded speed", "## Historical 114-page recorded speed")
        update = ("A later CPU deployment check on 12 fixed, randomly sampled pages is included below. "
            "It reruns all three engines and uses a validated ONNX configuration for Paddle. "
            "The historical full-book table remains a separate experiment.\n\n")
        first_end = article.index("\n") + 1
        article = article[:first_end] + "\n" + update + article[first_end:]
    if public_dataset:
        article = article.replace("Public annotated-document evaluation is still pending.", "An annotated text-region follow-up is included below.")
        article = article.replace("The download was blocked in the execution session, and no public-dataset results are claimed here.", "The completed annotated-region follow-up is reported below, separately from the book study.")
        public = json.loads((public_dataset / "publication-stats.json").read_text(encoding="utf-8"))
        section = ["## Annotated text-region follow-up", "",
            "These are dataset-provided text-region crops from OmniDocBench. Gold boundaries are supplied; this is a custom recognition test, not its official full-page score. Case and punctuation count. Possible annotation errors and training overlap remain limitations.", "",
            "| Engine | WER | CER | Regions/min | Failures, all passes |", "|---|---:|---:|---:|---:|"]
        for name, model in public["models"].items():
            section.append(f"| {name} | {model['wer']:.3%} | {model['cer']:.3%} | {model['pages_per_minute']:.2f} | {model['failures_all_passes']} |")
        section += ["", f"Dataset manifest SHA256: `{audit['public']['dataset_sha256']}`. Each engine used three measured passes. Paddle backend and resource scope are recorded in the accompanying results. These region timings must not be compared with full-book pages/min."]
        article += "\n\n" + "\n".join(section) + "\n"
    verification_html = ""
    if verification:
        v = verification
        section = ["## Repeated local Paddle configuration check", "",
            "A separate run used the unchanged three known-text control images, PP-OCRv6 medium weights and plain Paddle CPU backend with four requested threads. One first-image warmup preceded three measured passes. Every pass was scored and its normalized text hashes compared. This is a configuration check on three variants of one passage, not a full-book or diverse-document speed result.", "",
            "| Pass | Three-image time (s) | WER | CER |", "|---|---:|---:|---:|"]
        for i, (seconds, score) in enumerate(zip(v["pass_seconds"], v["quality_each_pass"]), 1):
            section.append(f"| {i} | {seconds:.3f} | {score['wer']:.3%} | {score['cer']:.3%} |")
        section += ["", f"Median three-image pass: {v['median_pass_seconds']:.3f} s; throughput: {v['pages_per_minute']:.3f} control images/min; pass-time CV: {v['pass_time_cv_percent']:.2f}%.",
            f"Identical normalized output across all passes: {v['identical_normalized_text_all_passes']}. All verification checks passed: {v['configuration_verified_on_control']}. Detector and recognizer weights were hashed and actual backend settings recorded.",
            f"Sampled peak process-tree RSS: {v['resources']['sampled_peak_tree_rss_mib']:.1f} MiB. Mean busy CPU cores over the entire worker: {v['resources']['mean_busy_cpu_cores_whole_worker']:.2f}. This scope includes imports, loading, warmup, all passes, scoring and writes; it is not inference-only CPU/RAM. Summed RSS can double-count shared pages.",
            "This plain-runtime check is limited to these controls. The subsequent ONNX comparison is reported separately below; the historical book table is unchanged."]
        article += "\n\n" + "\n".join(section) + "\n"
        verification_html = ("<h2>Repeated local Paddle check</h2><p>Configuration verified on control: <strong>"
            + str(v["configuration_verified_on_control"]) + "</strong>. Three measured passes after warmup; "
            + f"{v['pages_per_minute']:.3f} control images/min; pass-time CV {v['pass_time_cv_percent']:.2f}%. "
            + "This is the three-image synthetic control, separate from full-book throughput. "
            + '<a href="paddle-verification.md">Full verification report</a>.</p>')
    if onnx:
        article += ("\n\n## Validating the ONNX configuration\n\n"
            "The same cached PP-OCRv6 medium weights were converted to FP32 ONNX with no quantization. "
            "Source hashes matched the measured plain-Paddle control; both exported models passed full ONNX checks. "
            "The default ONNX diagnostic was also slow: its first measured image took 143.74 seconds before that diagnostic was stopped. "
            "It was not a completed three-pass result.\n\n"
            "A subsequent run enabled the documented [ONNX Runtime denormal-as-zero setting](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h). "
            "This changes how extremely small floating-point values are handled and can affect numerical outputs, so recognition quality was checked again. "
            f"Across three measured passes, median three-image time fell from {onnx['baseline_median_pass_seconds']:.3f} s with plain Paddle to {onnx['onnx_median_pass_seconds']:.3f} s with ONNX CPU and this setting. "
            f"The observed ratio was {onnx['observed_speed_ratio']:.2f}x on this control. Every normalized output matched the baseline, with zero character or word edits. "
            "That establishes a useful tested configuration on this small control; it does not establish maximum possible model speed or universal numerical equivalence.\n")
        quality_path = output / "onnx-book-quality-check.json"
        if quality_path.is_file():
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            article += (f"\nOn the later 12-page book subset ({quality['onnx_first_pass']['reference_words']:,} reference words), "
                f"every normalized first-pass ONNX output matched the saved historical native-Paddle output: {quality['all_12_normalized_outputs_match_historical_native']}. "
                f"Both produced {quality['onnx_first_pass']['word_errors']} word edits and {quality['onnx_first_pass']['character_errors']} character edits against the PDF text layer. "
                "The historical native run did not capture weight hashes, so this is output agreement with that saved run, rather than retrospective proof of its model identity.\n")
        verification_html += ('<h2>ONNX control comparison</h2><p>'
            + f"Observed control-set speed ratio: {onnx['observed_speed_ratio']:.2f}x. All outputs match baseline: {onnx['normalized_outputs_match_baseline']}. "
            + '<a href="onnx-comparison.md">ONNX comparison and configuration</a>.</p>')
    if deployment:
        fresh_rows = []
        for name, model in deployment["statistics"]["models"].items():
            fresh_rows.append([name + (" (ONNX)" if name == "paddleocr" else ""), f"{model['wer']:.3%}",
                f"{model['cer']:.3%}", f"{model['pages_per_minute']:.2f}", str(model["failures_all_passes"])])
        table_image(output / "deployment-results.png", "Fresh 12-page Windows CPU comparison",
            ["Engine", "WER", "CER", "Pages/min", "Failures"], fresh_rows,
            ["Reference: unverified PDF text layer. Three measured passes per engine; seed 20261018.",
             "Paddle: PP-OCRv6 medium / ONNX Runtime CPU / denormal-as-zero enabled.",
             "This is a sampled subset of one book; do not merge its timings with the historical table.",
             "Pass-time variation (CV): " + "; ".join(f"{k} {m['pass_time_cv_percent']:.2f}%" for k, m in deployment["statistics"]["models"].items())])
        source_csv = Path(deployment["dataset"]) / "page-stats.csv"
        (output / "deployment-page-stats.csv").write_bytes(source_csv.read_bytes())
        article += ("\n\n## Fresh three-engine deployment comparison\n\n"
            "All three engines were rerun sequentially on the same 12 saved 200-DPI RGB book pages. "
            "Page indices were selected by a fixed random seed before this run, without filtering on error scores. "
            "Each engine had one excluded warmup and three measured passes, with four inference threads requested. "
            "Paddle used the validated ONNX CPU configuration with denormal-as-zero enabled. "
            "These remain agreement scores against the unverified PDF text layer.\n\n"
            "![Fresh deployment results](deployment-results.png)\n\n"
            "| Engine | WER | CER | Pages/min | Failures, all passes |\n|---|---:|---:|---:|---:|\n"
            + "\n".join("| " + " | ".join(row) + " |" for row in fresh_rows)
            + "\n\nTesseract CLI startup is included; the Python models remain loaded. All first-pass predictions were rescored from their saved text. "
            "Every pass has numerical quality scores and normalized output hashes. Whole-worker CPU/RAM were sampled separately and include loading, warmup, scoring and writes. "
            "Those resource scopes differ from inference throughput.\n\n"
            f"Source PDF pages: {deployment['source_pdf_pages']}. Seed: 20261018. "
            "This current 12-page table and the historical 114-page table are different runs and must not be combined into a single speed ranking. "
            "The timing differences across runs also limit cross-session speed claims. These data do not establish a universal OCR winner.\n")
        repeatability = ["\n\n### Timing repeatability and resource scope", "",
            "| Engine | Pass range (s) | Pass-time CV | Peak sampled tree RSS (MiB) |", "|---|---:|---:|---:|"]
        for name, model in deployment["statistics"]["models"].items():
            lo, hi = model["pass_time_range_seconds"]
            repeatability.append(f"| {name} | {lo:.3f}–{hi:.3f} | {model['pass_time_cv_percent']:.2f}% | {deployment['resources'][name]['sampled_peak_tree_rss_mib']:.1f} |")
        repeatability += ["", "Indra's first pass was materially faster than its later passes in this run. That timing variation remains unexplained; the median is reported with the range and CV, and should not be presented as a stable universal speed estimate. CPU/RAM cover the whole worker, including warmup and scoring. Throughput counts image-to-text calls only."]
        article += "\n".join(repeatability) + "\n"
        md_path = output / "deployment-comparison.md"
        md_text = md_path.read_text(encoding="utf-8")
        if "### Timing repeatability and resource scope" not in md_text:
            md_path.write_text(md_text + "\n" + "\n".join(repeatability) + "\n", encoding="utf-8")
        verification_html += '<h2>Fresh 12-page comparison</h2><p><a href="deployment-comparison.md">Measured results and resources</a></p><img src="deployment-results.png" alt="Fresh 12-page three-engine CPU comparison">'
    fullbook_html = ""
    if fullbook:
        article, fullbook_html = fullbook_article(fullbook, audit["fresh_fullbook"], onnx, public_dataset)
    cpu_html = ""
    if cpu:
        article = (output / "cpu-four-comparison.md").read_text(encoding="utf-8")
        article += "\n## Paired word-error differences\n\nNegative differences favor the left engine. These are descriptive within-book page-bootstrap intervals (5,000 resamples, seed 20261018), not generalization estimates across independent books.\n\n| Comparison | WER difference (pp) | Paired 95% interval (pp) |\n|---|---:|---:|\n"
        for pair in cpu["statistics"]["paired_comparisons"]:
            lo, hi = pair["wer_difference_95"]
            article += f"| {pair['left']} minus {pair['right']} | {pair['wer_difference_percentage_points']:.4f} | {lo:.4f} to {hi:.4f} |\n"
        article += ("\n## Checks and limits\n\nAll 456 first-pass predictions were rescored from their saved text. Every measured pass has numerical quality scores and normalized output hashes; all engines produced identical normalized output across their three current passes. Runtime/model hashes and source-image/reference hashes are retained. CPU/RAM include the whole worker; engine setup is not true cold-start latency, and Tesseract loads its OCR model inside each timed CLI call.\n\n"
            "Indra returned to roughly 207 pages/min in this run after a previous 114-page run measured 32.38 pages/min with matching model hashes. The cross-session drift remains unexplained. Keep historical measurements separate; stable pass times here do not establish universal speed. The earlier medium-Paddle comparison is preserved in fullbook-comparison.md. Tiny and medium results must not be presented as one model configuration.\n\n"
            "Current known-text single-image probes: Indra and Tesseract had zero word/character edits; EasyOCR had 19 word and 37 character edits; Tiny had 3 word and 2 character edits. These functional probes are not a diverse ground-truth benchmark. No independently annotated real-document follow-up has completed.\n\n"
            "MEDIUM_POST.md is the reader-facing four-engine draft. The commercial PDF, source images, reference text and raw book predictions stay out of the release archive. Nothing has been posted to Medium.\n")
        cpu_html = '<h2>Current four-engine CPU comparison</h2><p><a href="cpu-four-comparison.md">Results, initialization and RAM</a> · <a href="medium-post.html">Updated Medium post preview</a></p><img src="cpu-four-results.png" alt="Current Indra, Tesseract, EasyOCR and PaddleOCR Tiny comparison">'
    if transcription_file.is_file():
        article += "\n## Supplied transcription follow-up\n\n[Candidate-reference comparison](transcription-comparison.md) includes all 114 pages, a 99-page sensitivity analysis excluding all flagged pages, and quote-style normalization checks. The supplied transcription is not independently verified ground truth; original PDF-reference scores above are preserved. Speed and RAM were not remeasured.\n"
        cpu_html += '<h2>Candidate-transcription follow-up</h2><p><a href="transcription-comparison.md">New reference comparison and sensitivity checks</a>. The supplied transcription is not independently verified ground truth. Original reference scores and timing measurements remain separate.</p>'
    (output / "ARTICLE.md").write_text(article, encoding="utf-8")
    table_rows = "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in row) + "</tr>" for row in rows)
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>OCR publication package</title>
<style>body{{font:18px/1.6 system-ui;max-width:1050px;margin:48px auto;padding:0 24px;color:#16252c}}table{{border-collapse:collapse;width:100%}}td,th{{padding:14px;border-bottom:1px solid #ccd5d8;text-align:left}}.notice{{background:#fff1c9;padding:18px}}img{{max-width:100%}}a{{color:#075c83}}</style>
<h1>Indra, Tesseract and PaddleOCR on a Windows CPU</h1>
<p>114 identical RGB page images at 200 DPI. Three measured passes per engine, one warmup excluded. All 342 saved first-pass predictions have been rescored from their text files.</p>
<p class="notice">This package supports a Windows case study. The PDF text reference is unverified. The 114-page table is historical; current ONNX and deployment measurements are separate below. Public annotated results: {audit['public_annotated_results']}.</p>
{cpu_html}
{fullbook_html}
<h2>Historical native-runtime comparison</h2>
<table><thead><tr><th>Engine</th><th>WER</th><th>CER</th><th>Pages/min</th><th>Failures</th></tr></thead><tbody>{table_rows}</tbody></table>
<p>WER/CER count edit distance relative to the reference. Do not convert 100−WER into a probability of correct recognition. Tesseract CLI startup is included; Python models remain resident.</p>
<p>All three engines made zero word or character edits on a known-text synthetic control: 372 repeated words across three variants of one 124-word passage. This small clean-print control does not establish accuracy on real scans.</p>
<h2>Paddle diagnosis</h2><p>On the same control image, the original oneDNN configuration took 164.61 seconds. Plain CPU took 109.83 seconds; plain CPU with legacy IR took 110.56 seconds. Each completed with zero word edits. These individual cold runs are separate from the full-book timings. Single-thread oneDNN was stopped while still detecting after 136 seconds.</p>
{verification_html}
<h2>Files</h2><ul><li><a href="ARTICLE.md">Medium article</a></li><li><a href="book-results.png">Medium table image</a></li><li><a href="book-results.csv">Results CSV</a></li><li><a href="evidence-audit.json">Machine-readable evidence audit</a></li></ul>
<p>Nothing has been posted to Medium. The source commercial book and its text/images are excluded from the shareable archive.</p></html>'''
    (output / "index.html").write_text(report, encoding="utf-8")
    (output / "README.md").write_text(
        "# Publication package\n\nARTICLE.md is the article; book-results.png is its table image. index.html is the local overview. evidence-audit.json records rescoring and file hashes.\n\n"
        "The reader-facing short visual edition is medium-post.html / MEDIUM_POST.md, with chart-speed.png, chart-word-errors.png and chart-memory.png. MEDIUM_FULL_METHODS.md preserves the detailed story.\n\n"
        f"Ready for editorial review as a narrowly scoped Windows case study. Current four-engine results included: {bool(cpu)}; MEDIUM_POST.md / medium-post.html are the reader-facing draft. Public annotated results included: {bool(public_dataset)}. ONNX control comparison included: {bool(onnx)}. Fresh three-engine subset included: {bool(deployment)}. Fresh 114-page comparison included: {bool(fullbook)}. These establish tested configurations and observed results, not universal accuracy or maximum possible speed. No publication has occurred.\n\n"
        "To complete the annotated follow-up in a network-enabled terminal, run the project-root command:\n\n"
        "```powershell\n.\\.benchmark-venv\\Scripts\\python.exe finish_publication.py\n```\n\n"
        "It downloads the fixed-seed annotated subset, runs three passes per engine, collects resources, audits predictions, and rebuilds this package. Any download, timeout or audit failure stops completion and is recorded.\n",
        encoding="utf-8")
    print(f"Built {output / 'index.html'}; audited {sum(e['pages_recomputed'] for e in book_audit['engines'].values())} book predictions.")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-dataset", type=Path)
    args = parser.parse_args()
    build(args.public_dataset)
