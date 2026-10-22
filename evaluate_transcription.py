"""Rescore saved OCR against a supplied candidate transcription; never rerun OCR."""

from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import csv
import hashlib
import html
import json
import random
import re

from benchmark_stats import interval, rate_samples
from evaluate import aggregate, compare, normalize, sha256


ROOT = Path(__file__).resolve().parent
STATUS = "AI-assisted candidate transcription; not independently verified ground truth"


def fold_quotes(text):
    return text.translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"'}))


def parse_transcription(text, count=114):
    headings = re.findall(r"^## (page_\d{3}\.txt)[ \t]*$", text, re.M)
    expected = [f"page_{n:03}.txt" for n in range(1, count + 1)]
    if headings != expected:
        raise ValueError("Transcription must contain each physical page exactly once, in order")
    blocks = re.findall(r"^## (page_\d{3}\.txt)[ \t]*\n\s*```text[ \t]*\n(.*?)^```[ \t]*$", text, re.M | re.S)
    if [name for name, _ in blocks] != expected:
        raise ValueError("Each page needs one complete fenced text block")
    if text.count("```") != count * 2:
        raise ValueError("Unexpected or unclosed code fence")
    if text.count("## Uncertain passages") != 1:
        raise ValueError("Expected an explicit uncertainty-notes section")
    notes = text.split("## Uncertain passages", 1)[1]
    flagged = {int(n) for n in re.findall(r"^- page_(\d{3})\.txt", notes, re.M)}
    pages, inline = {}, {}
    for name, raw in blocks:
        page = int(name[5:8])
        if not raw.strip() or raw.count(f"Page {page}/{count}") != 1:
            raise ValueError(f"Missing/incorrect physical page footer: {name}")
        guesses = re.findall(r"\[UNCERTAIN: ([^\]\n]+)\]", raw)
        cleaned = re.sub(r"\[UNCERTAIN: ([^\]\n]+)\]", lambda m: m[1], raw)
        if "[UNCERTAIN" in cleaned:
            raise ValueError(f"Malformed uncertainty marker: {name}")
        if guesses:
            flagged.add(page)
            inline[page] = len(guesses)
        pages[page] = {"raw": raw, "best_reading": cleaned}
    if not flagged.issubset(pages):
        raise ValueError("Uncertainty notes refer to an absent page")
    return pages, sorted(flagged), inline


def pooled(pages):
    return aggregate(pages, "score")


def summarize(scores, selected):
    selected_scores = {engine: [p for p in pages if p["page"] in selected]
                       for engine, pages in scores.items()}
    totals = {engine: pooled(pages) for engine, pages in selected_scores.items()}
    rng = random.Random(20261022)
    samples = [rng.choices(range(len(selected)), k=len(selected)) for _ in range(5000)]
    rates = {engine: rate_samples([p["score"]["word_errors"] for p in pages],
                [p["score"]["reference_words"] for p in pages], samples)
             for engine, pages in selected_scores.items()}
    pairs = []
    for left, right in combinations(scores, 2):
        pairs.append({"left": left, "right": right,
            "wer_difference_percentage_points": (totals[left]["wer"] - totals[right]["wer"]) * 100,
            "paired_95_interval_percentage_points": interval([(a-b)*100 for a,b in zip(rates[left],rates[right])])})
    return {"pages": len(selected), "page_ids": sorted(selected), "engines": totals,
            "paired_wer_differences": pairs,
            "uncertainty_method": "5000 paired page bootstrap resamples, seed 20261022; descriptive within this book, not reference correctness or generalization"}


def main():
    source = ROOT / "Rich_Dad_Poor_Dad_transcription.md"
    original_hash = sha256(source)
    pages, flagged, inline = parse_transcription(source.read_text(encoding="utf-8-sig"))
    completed = ROOT / "publication/cpu-four-comparison.json"
    saved = json.loads(completed.read_text(encoding="utf-8"))
    if not saved["completed"] or set(saved["audit"]["engines"]) != {"indra", "tesseract", "easyocr", "paddleocr_tiny"}:
        raise ValueError("Need the completed four-engine source experiment")
    dataset = (ROOT / saved["dataset"]).resolve()
    if not dataset.is_relative_to(ROOT / "eval/benchmarks/cpu-four-114"):
        raise ValueError("Unexpected source experiment directory")
    manifest = json.loads((dataset / "dataset.json").read_text(encoding="utf-8"))
    if sha256(dataset / "dataset.json") != saved["audit"]["dataset_sha256"]:
        raise ValueError("Source dataset manifest changed")
    if sha256(ROOT / "Rich_Dad_Poor_Dad.pdf") != manifest["pdf_sha256"]:
        raise ValueError("Source PDF changed")
    if [p["page"] for p in manifest["pages"]] != list(pages):
        raise ValueError("Transcription and source pages do not match")
    layer_scores = []
    for item in manifest["pages"]:
        for field in ("image", "reference"):
            if sha256(dataset / item[field]) != item[field + "_sha256"]:
                raise ValueError(f"Source input changed: {item[field]}")
        layer = (dataset / item["reference"]).read_text(encoding="utf-8")
        layer_scores.append({"page": item["page"], "score": compare(normalize(pages[item["page"]]["best_reading"], []), normalize(layer, []))})
    folder = ROOT / "eval/transcription" / original_hash[:16]
    folder.mkdir(parents=True, exist_ok=True)
    references = folder / "reference"
    references.mkdir(exist_ok=True)
    for page, text in pages.items():
        (references / f"page_{page:03}.txt").write_text(text["raw"], encoding="utf-8")
    scores, hashes, quote_scores = {}, {}, {}
    for engine, audit in saved["audit"]["engines"].items():
        result_file = Path(audit["result_file"]).resolve()
        if not result_file.is_relative_to(dataset) or sha256(result_file) != audit["result_sha256"]:
            raise ValueError(f"Changed/outside saved result: {engine}")
        if audit["execution_failures_all_passes"] or not audit["all_pass_outputs_match_saved_first_pass"]:
            raise ValueError(f"Source passes are incomplete or unstable: {engine}")
        scores[engine], hashes[engine] = [], {}
        quote_scores[engine] = []
        for page in pages:
            name = f"page_{page:03}.txt"
            prediction = result_file.parent / name
            digest = sha256(prediction)
            if digest != audit["prediction_sha256"][name]:
                raise ValueError(f"Changed prediction: {engine}/{name}")
            hashes[engine][name] = digest
            reference = normalize(pages[page]["best_reading"], [])
            prediction_text = normalize(prediction.read_text(encoding="utf-8"), [])
            score = compare(reference, prediction_text)
            scores[engine].append({"page": page, "score": score})
            quote_scores[engine].append({"page": page, "score": compare(fold_quotes(reference), fold_quotes(prediction_text))})
    if sha256(source) != original_hash:
        raise ValueError("Transcription changed during evaluation")
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "reference_status": STATUS,
        "source_file": source.name, "source_sha256": original_hash, "evaluator_sha256": sha256(Path(__file__)),
        "source_comparison_sha256": sha256(completed), "source_dataset": saved["dataset"],
        "flagged_pages": flagged, "inline_uncertainties_by_page": inline,
        "scoring_policy": "Same NFC/whitespace/Markdown-heading normalization as prior experiment; preserve case/punctuation. For provisional full-book scores only, replace six UNCERTAIN wrappers with their supplied best readings. Do not treat uncertainty markup as printed text. No text silently corrected.",
        "visual_check_scope": "Spot-checked saved page images 2, 52 and 76; no complete independent visual verification. Page 52 contains visibly corrupted amount glyphs. On page 76 the candidate uses straight apostrophes in printed HTML attributes where the image shows curly apostrophes, demonstrating that even unflagged pages need review. Original transcription left unchanged. Transcriber model/version and generation records were not supplied.",
        "full_book_provisional": summarize(scores, set(pages)),
        "excluding_all_flagged_pages": summarize(scores, set(pages) - set(flagged)),
        "quote_folded_sensitivity": {"policy": "Additionally map U+2018/U+2019 to ASCII apostrophe and U+201C/U+201D to ASCII double quote, on reference and prediction; preserve case and other punctuation. Supplemental analysis only, not a replacement for the primary strict scores.",
            "full_book": {engine: pooled(p) for engine, p in quote_scores.items()},
            "excluding_flagged_pages": {engine: pooled([p for p in values if p['page'] not in flagged]) for engine, values in quote_scores.items()}},
        "candidate_vs_pdf_layer": {"overall": pooled(layer_scores), "exact_normalized_pages": sum(p["score"]["character_errors"] == 0 for p in layer_scores)},
        "timing": "No OCR rerun. Speed/RAM remain the original four-engine experiment, not new measurements.",
        "prediction_sha256": hashes}
    write_outputs(folder, report, scores, layer_scores)
    print((ROOT / "publication/transcription-comparison.md").read_text(encoding="utf-8"))


def write_outputs(folder, report, scores, layer_scores):
    (folder / "word-differences.json").write_text(json.dumps({"engines": scores, "pdf_layer": layer_scores}, ensure_ascii=False), encoding="utf-8")
    rows = []
    for engine, pages in scores.items():
        for page in pages:
            for change in page["score"]["changes"]:
                values = [engine, page["page"], "flagged" if page["page"] in report["flagged_pages"] else "", change["type"], change["reference"], change["prediction"], change["context"]]
                rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>")
    (folder / "report.html").write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>Candidate transcription: word differences</title>
<style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ccc;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#eee}input{padding:10px;width:80%}</style>
<h1>Candidate transcription: word differences</h1><p>Not independently verified ground truth. Full-book scores use supplied best readings in uncertain passages; flagged pages are labeled. Missing/Added are relative to the candidate transcription. Book text stays local.</p>
<input id="q" aria-label="Filter changes" placeholder="Filter by engine, page, word or flagged"><p id="count"></p><table><thead><tr><th>Engine</th><th>Page</th><th>Flag</th><th>Change</th><th>Candidate</th><th>OCR output</th><th>Candidate context</th></tr></thead><tbody>''' + "".join(rows) + '''</tbody></table><script>const rows=[...document.querySelectorAll('tbody tr')];function filter(){const q=document.querySelector('#q').value.toLowerCase();let n=0;rows.forEach(r=>{r.hidden=!r.textContent.toLowerCase().includes(q);if(!r.hidden)n++});document.querySelector('#count').textContent=n+' differences shown'}document.querySelector('#q').addEventListener('input',filter);filter()</script></html>''', encoding="utf-8")
    public = ROOT / "publication"
    (public / "transcription-comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (public / "transcription-page-stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["engine", "page", "flagged", "reference_words", "reference_characters", "word_errors", "character_errors", "wer", "cer"])
        writer.writeheader()
        for engine, pages in scores.items():
            for page in pages:
                writer.writerow({"engine": engine, "page": page["page"], "flagged": page["page"] in report["flagged_pages"], **{k: page["score"][k] for k in writer.fieldnames[3:]}})
    lines = ["# Evaluation against supplied transcription", "", STATUS + ".", "",
        f"All 114 pages parsed. {len(report['flagged_pages'])} pages flagged in the supplied uncertainty notes; six inline uncertainty markers. The original file is unchanged.", "",
        "Full-book scores use the supplied best readings in uncertainty markers. The second analysis excludes every flagged page for every engine; this is a sensitivity check, not a verified subset.", ""]
    for key, title in [("full_book_provisional", "114-page provisional comparison"), ("excluding_all_flagged_pages", "Excluding all flagged pages")]:
        section = report[key]
        lines += [f"## {title} ({section['pages']} pages)", "", "| Engine | WER | CER | Word edits | Reference words |", "|---|---:|---:|---:|---:|"]
        for engine, score in section["engines"].items():
            lines.append(f"| {engine} | {score['wer']:.3%} | {score['cer']:.3%} | {score['word_errors']} | {score['reference_words']} |")
        lines += ["", "Paired WER differences (percentage points; negative favors left). Descriptive 95% page-bootstrap intervals within this book:", ""]
        for pair in section["paired_wer_differences"]:
            lo, hi = pair["paired_95_interval_percentage_points"]
            lines.append(f"- {pair['left']} vs {pair['right']}: {pair['wer_difference_percentage_points']:.3f} [{lo:.3f}, {hi:.3f}]")
        lines.append("")
    lines += ["## Quote-style sensitivity", "", report["quote_folded_sensitivity"]["policy"], "",
        "| Engine | 114-page WER | 114-page CER | 99-page WER | 99-page CER |", "|---|---:|---:|---:|---:|"]
    for engine, score in report["quote_folded_sensitivity"]["full_book"].items():
        subset = report["quote_folded_sensitivity"]["excluding_flagged_pages"][engine]
        lines.append(f"| {engine} | {score['wer']:.3%} | {score['cer']:.3%} | {subset['wer']:.3%} | {subset['cer']:.3%} |")
    lines.append("")
    layer = report["candidate_vs_pdf_layer"]
    lines += [f"Candidate vs embedded PDF text: {layer['overall']['wer']:.3%} WER, {layer['overall']['cer']:.3%} CER; {layer['exact_normalized_pages']}/114 exact normalized pages. Agreement cannot establish independent provenance or correctness.", "",
        "Flagged pages: " + ", ".join(map(str, report["flagged_pages"])) + ".", "",
        report["visual_check_scope"], "", report["timing"], "",
        "The Medium draft retains the original reference scores. This supplemental candidate-reference analysis must not be presented as verified accuracy.", "",
        f"The word-level book-text comparison stays local at `{(folder / 'report.html').relative_to(ROOT).as_posix()}` and is excluded from Git.", "",
        f"Transcription SHA256: `{report['source_sha256']}`", ""]
    (public / "transcription-comparison.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
