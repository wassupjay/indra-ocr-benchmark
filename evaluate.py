"""Compare saved OCR pages with the PDF text layer; never reruns OCR."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import unicodedata
from datetime import datetime, timezone
from importlib.metadata import version

from rapidfuzz.distance import Levenshtein


def normalize(text, exclude_lines):
    text = unicodedata.normalize("NFC", text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        if any(re.fullmatch(pattern, line.strip()) for pattern in exclude_lines):
            continue
        lines.append(line)
    # Preserve case, punctuation, numbers and hyphens: these can be OCR errors.
    return " ".join("\n".join(lines).split())


def compare(reference, prediction):
    expected, actual = reference.split(), prediction.split()
    changes = []
    counts = {"replace": 0, "delete": 0, "insert": 0}
    for tag, a, b, c, d in Levenshtein.opcodes(expected, actual):
        if tag == "equal":
            continue
        counts[tag] += max(b - a, d - c)
        changes.append({
            "type": {"replace": "Changed", "delete": "Missing", "insert": "Added"}[tag],
            "reference": " ".join(expected[a:b]),
            "prediction": " ".join(actual[c:d]),
            "context": " ".join(expected[max(0, a - 5):min(len(expected), b + 5)]),
        })
    word_errors = Levenshtein.distance(expected, actual)
    char_errors = Levenshtein.distance(reference, prediction)
    return {
        "reference_words": len(expected), "reference_characters": len(reference),
        "word_errors": word_errors, "character_errors": char_errors,
        "wer": word_errors / len(expected) if expected else None,
        "cer": char_errors / len(reference) if reference else None,
        "counts": counts, "changes": changes,
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_pdftotext():
    found = shutil.which("pdftotext")
    if found:
        return found
    packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WinGet/Packages"
    matches = sorted(packages.glob("oschwartz10612.Poppler_*/**/pdftotext.exe"))
    if matches:
        return str(matches[-1])
    raise ValueError("pdftotext not found. Add Poppler's bin directory to PATH.")


def percent(value):
    return "Unscorable" if value is None else f"{value:.2%}"


def write_report(folder, result):
    escape = lambda value: html.escape(str(value))
    page_rows, change_rows = [], []
    for page in result["pages"]:
        score = page["normalized"]
        page_rows.append(
            f'<tr><td>{page["page"]}</td><td>{score["reference_words"]}</td>'
            f'<td>{percent(score["wer"])}</td><td>{percent(score["cer"])}</td>'
            f'<td>{percent(page["raw"]["wer"])}</td>'
            f'<td>{len(score["changes"])}</td></tr>'
        )
        for change in score["changes"]:
            change_rows.append("<tr>" + "".join(
                f"<td>{escape(value)}</td>" for value in (
                    page["page"], change["type"], change["reference"],
                    change["prediction"], change["context"]
                )
            ) + "</tr>")
    overall = result["overall"]
    worst = sorted(
        (p for p in result["pages"] if p["normalized"]["wer"] is not None),
        key=lambda p: p["normalized"]["wer"], reverse=True
    )[:10]
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Indra word comparison</title><style>
body{{font:16px system-ui;margin:32px;color:#17212b}}table{{border-collapse:collapse;width:100%;margin:16px 0}}
th,td{{border:1px solid #ccd3db;padding:8px;text-align:left;vertical-align:top;white-space:pre-wrap}}
th{{background:#edf2f7;position:sticky;top:0}}input{{padding:10px;width: min(700px,90%)}}
.notice{{background:#fff3cd;padding:16px}}td:nth-child(4){{background:#f4f8fc}}
</style><h1>Indra word comparison</h1>
<p class="notice">Candidate reference: PDF text layer, not independently verified ground truth.
Scores measure agreement with that layer. Layout, tables and reading order are not scored separately.</p>
<p>{len(result['pages'])} pages · normalized WER <b>{percent(overall['wer'])}</b> ·
normalized CER <b>{percent(overall['cer'])}</b> · {overall['word_errors']} word edits /
{overall['reference_words']} reference words</p>
<p>Case and punctuation count. Unicode NFC, whitespace and heading markers are normalized.
Excluded full lines: {escape(json.dumps(result['metadata']['exclude_lines'], ensure_ascii=False))}.</p>
<p>Raw WER: {percent(result['raw_overall']['wer'])}.
Empty-reference pages: {escape(result['metadata']['empty_reference_pages'])}.
Evaluation time: {result['metadata']['evaluation_seconds']:.2f}s.</p>
<p class="notice">Reference quality flags: {escape(result['metadata']['reference_quality_flags'])}</p>
<p>Worst pages: {escape(', '.join(str(p['page']) + ' (' + percent(p['normalized']['wer']) + ')' for p in worst))}</p>
<h2>Pages</h2><table><thead><tr><th>PDF page</th><th>Reference words</th><th>WER</th>
<th>CER</th><th>Raw WER</th><th>Change groups</th></tr></thead><tbody>{''.join(page_rows)}</tbody></table>
<h2>Word changes</h2><p>Missing = in reference only. Added = in Indra only.
Changed = substitution. Context comes from the reference.</p>
<input id="search" placeholder="Filter by page, word, or error type" aria-label="Filter word changes">
<p id="count"></p><table><thead><tr><th>PDF page</th><th>Type</th><th>Reference</th>
<th>Indra</th><th>Reference context</th></tr></thead><tbody id="changes">{''.join(change_rows)}</tbody></table>
<script>const rows=[...document.querySelectorAll('#changes tr')];
function filter(){{const q=document.querySelector('#search').value.toLowerCase();let n=0;
rows.forEach(r=>{{r.hidden=!r.textContent.toLowerCase().includes(q);if(!r.hidden)n++}});
document.querySelector('#count').textContent=n+' change groups shown';}}
document.querySelector('#search').addEventListener('input',filter);filter();</script></html>"""
    (folder / "report.html").write_text(document, encoding="utf-8")
    (folder / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


def aggregate(pages, key):
    totals = {name: sum(p[key][name] for p in pages) for name in (
        "reference_words", "reference_characters", "word_errors", "character_errors"
    )}
    totals["wer"] = totals["word_errors"] / totals["reference_words"] if totals["reference_words"] else None
    totals["cer"] = totals["character_errors"] / totals["reference_characters"] if totals["reference_characters"] else None
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=Path("Rich_Dad_Poor_Dad.pdf"))
    parser.add_argument("--predictions", type=Path, default=Path("eval/prediction"))
    parser.add_argument("--config", type=Path, default=Path("eval/config.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        excluded = config["exclude_lines"]
        for pattern in excluded:
            re.compile(pattern)
        executable = find_pdftotext()
        extracted = subprocess.run(
            [executable, "-layout", "-enc", "UTF-8", str(args.pdf), "-"],
            check=True, capture_output=True
        ).stdout.decode("utf-8")
        references = extracted.split("\f")
        if references[-1].strip() == "":
            references.pop()  # Poppler terminates its last page with form feed.
        if not references or not any(text.strip() for text in references):
            raise ValueError("PDF has no usable text layer. Use a labeled benchmark or verified transcription.")
        expected = {f"page_{n:03}.txt" for n in range(1, len(references) + 1)}
        available = {p.name for p in args.predictions.glob("page_*.txt")}
        if expected != available:
            raise ValueError(f"Page mismatch. Missing: {sorted(expected - available)}; extra: {sorted(available - expected)}")
        ocr_metadata_path = args.predictions / "run.json"
        ocr_metadata = json.loads(ocr_metadata_path.read_text(encoding="utf-8")) if ocr_metadata_path.exists() else None
        if ocr_metadata and ocr_metadata.get("pdf_sha256") != sha256(args.pdf):
            raise ValueError("Saved OCR was produced from a different PDF (hash mismatch).")
        folder = Path("eval/runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        folder.mkdir(parents=True)
        for name in ("reference", "prediction", "normalized/reference", "normalized/prediction"):
            (folder / name).mkdir(parents=True)
        pages = []
        for number, reference in enumerate(references, start=1):
            filename = f"page_{number:03}.txt"
            prediction_path = args.predictions / filename
            prediction = prediction_path.read_text(encoding="utf-8-sig")
            shutil.copyfile(prediction_path, folder / "prediction" / filename)
            ref, pred = normalize(reference, excluded), normalize(prediction, excluded)
            (folder / "reference" / filename).write_text(reference, encoding="utf-8")
            (folder / "normalized/reference" / filename).write_text(ref, encoding="utf-8")
            (folder / "normalized/prediction" / filename).write_text(pred, encoding="utf-8")
            raw = compare(reference, prediction)
            raw.pop("changes")
            pages.append({"page": number, "prediction_sha256": sha256(prediction_path),
                          "raw": raw, "normalized": compare(ref, pred)})
        metadata = {
            "reference_status": "unverified PDF text layer", "pdf": str(args.pdf.resolve()),
            "reference_quality_flags": {
                "pages_with_replacement_characters": [i for i, text in enumerate(references, 1) if "\ufffd" in text],
                "pages_under_20_words": [i for i, text in enumerate(references, 1) if len(text.split()) < 20],
            },
            "pdf_sha256": sha256(args.pdf), "exclude_lines": excluded,
            "normalization": "NFC; collapse whitespace; strip Markdown heading prefixes; preserve case and punctuation",
            "metric": "Levenshtein edits / reference length; Unicode code points for CER; whitespace tokens for WER",
            "rapidfuzz_version": version("rapidfuzz"),
            "evaluator_sha256": sha256(Path(__file__)),
            "pdftotext_version": subprocess.run([executable, "-v"], capture_output=True, text=True).stderr.strip(),
            "evaluation_seconds": time.perf_counter() - started,
            "empty_reference_pages": [p["page"] for p in pages if not p["normalized"]["reference_words"]],
            "ocr_run": ocr_metadata or "Not recorded for these existing predictions",
        }
        result = {"metadata": metadata, "overall": aggregate(pages, "normalized"),
                  "raw_overall": aggregate(pages, "raw"), "pages": pages}
        write_report(folder, result)
        print(f"Candidate-reference comparison: {len(pages)} pages")
        print(f"WER {percent(result['overall']['wer'])}; CER {percent(result['overall']['cer'])}")
        print(f"Report: {(folder / 'report.html').resolve()}")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
