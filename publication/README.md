# Publication package

ARTICLE.md is the article; book-results.png is its table image. index.html is the local overview. evidence-audit.json records rescoring and file hashes.

Ready for editorial review as a narrowly scoped Windows case study. Current four-engine results included: True; MEDIUM_POST.md / medium-post.html are the reader-facing draft. Public annotated results included: False. ONNX control comparison included: True. Fresh three-engine subset included: True. Fresh 114-page comparison included: True. These establish tested configurations and observed results, not universal accuracy or maximum possible speed. No publication has occurred.

To complete the annotated follow-up in a network-enabled terminal, run the project-root command:

```powershell
.\.benchmark-venv\Scripts\python.exe finish_publication.py
```

It downloads the fixed-seed annotated subset, runs three passes per engine, collects resources, audits predictions, and rebuilds this package. Any download, timeout or audit failure stops completion and is recorded.

Current reader-facing edition: MEDIUM_POST.md / medium-post.html, a short body plus three charts (speed, two-reference word error rates, RAM). Full previous story preserved in MEDIUM_FULL_METHODS.md. Numerical sources: chart-data.json.
