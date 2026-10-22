Run from this project folder:

```powershell
python evaluate.py
```

This reuses `eval/prediction/page_NNN.txt`; it does not run Indra.
It extracts the PDF text layer automatically, checks that page filenames match,
and creates a timestamped folder under `eval/runs` with:

- `report.html`: searchable word changes, per-page scores and worst pages.
- `results.json`: scores, error counts, versions, hashes and comparison rules.
- Original and normalized reference/prediction files for reproducing the comparison.

Open the latest report:

```powershell
Get-ChildItem eval/runs/*/report.html | Sort-Object FullName | Select-Object -Last 1 | Invoke-Item
```

WER is word Levenshtein edits divided by reference words. CER uses Unicode code
points. Overall scores sum edits and reference lengths across pages; they do not
average page percentages. Both rates can exceed 100%. Empty-reference pages have
no individual rate, but their inserted text contributes to overall error counts.

Raw scores include formatting. Normalized scores collapse whitespace, use Unicode
NFC, remove Markdown heading prefixes, and exclude the full lines configured in
`eval/config.json` from both inputs. Case, punctuation, numbers and hyphens remain
significant. The supplied exclusions cover this PDF's title/author header and
page/website footer. Disable exclusions with an empty list if those are required
content. Inspect normalized files when reviewing scores.

These scores measure agreement with an **unverified PDF text layer**. Spot-check
its text and reading order against a small representative sample before treating
it as ground truth. Garbled or sparse text layers require a labeled benchmark or
verified transcription. A second OCR engine does not establish ground truth.
This evaluator does not measure table structure or layout separately.

`python demo.py` reruns OCR and replaces `extracted.md` and matching prediction
files. Future runs record Indra version, threads, PDF hash, runtime and throughput
in `eval/prediction/run.json`. Historical runs have no recorded OCR timing.
Page numbers use Indra's PDF page index plus one. Evaluation rejects missing or extra pages.

Keep the same reference, rules and test pages when comparing OCR settings.
Use separate development pages for tuning and leave final test pages untouched.
Improving comparison rules does not itself improve OCR recognition.

Dependencies: Poppler's `pdftotext` and `rapidfuzz` (already installed here).
Checks: `python -m unittest test_evaluate.py`.
