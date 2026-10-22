"""Image-only OCR comparison. No engine receives the source PDF."""

import argparse
from datetime import datetime, timezone
import html
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

from evaluate import aggregate, compare, find_pdftotext, normalize, percent, sha256


ENGINES = ("indra", "tesseract", "paddleocr")
ADDITIONAL_ENGINES = ("easyocr", "paddleocr_tiny")


def dataset_engines(manifest):
    engines = manifest.get("engines", ENGINES)
    if (not isinstance(engines, (list, tuple)) or not engines
        or any(e not in (*ENGINES, *ADDITIONAL_ENGINES) for e in engines)
        or len(set(engines)) != len(engines)):
        raise ValueError("Dataset engines must be distinct supported engine names")
    return tuple(engines)


def load_dataset(folder):
    manifest = json.loads((folder / "dataset.json").read_text(encoding="utf-8"))
    dataset_engines(manifest)
    root = folder.resolve()
    if not manifest.get("pages"):
        raise ValueError("Dataset has no images.")
    for number, page in enumerate(manifest["pages"], 1):
        if page["page"] != number:
            raise ValueError("Dataset item IDs must be sequential starting at 1.")
        for key in ("image", "reference"):
            path = (root / page[key]).resolve()
            if path.parent != root:
                raise ValueError("Dataset file must be directly inside its dataset directory.")
            if sha256(path) != page[f"{key}_sha256"]:
                raise ValueError(f"Dataset file changed: {page[key]}")
    return manifest


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def tesseract_path():
    found = shutil.which("tesseract")
    if found:
        return found
    for base in (Path("C:/Program Files"), Path(os.environ.get("LOCALAPPDATA", "")) / "Programs"):
        candidate = base / "Tesseract-OCR/tesseract.exe"
        if candidate.exists():
            return str(candidate)
    raise RuntimeError("Tesseract is not installed or not on PATH.")


def prepare(pdf, dpi, limit):
    import pypdfium2 as pdfium

    digest = sha256(pdf)
    folder = Path("eval/benchmarks") / f"{digest[:12]}-{dpi}dpi-{limit or 'all'}"
    manifest_path = folder / "dataset.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for page in manifest["pages"]:
            for kind in ("image", "reference"):
                if sha256(folder / page[kind]) != page[f"{kind}_sha256"]:
                    raise RuntimeError(f"Dataset file changed: {page[kind]}")
        return folder, manifest
    folder.mkdir(parents=True, exist_ok=True)
    executable = find_pdftotext()
    command = [executable, "-layout", "-enc", "UTF-8"]
    if limit:
        command += ["-l", str(limit)]
    raw = subprocess.run(command + [str(pdf), "-"], check=True, capture_output=True).stdout.decode("utf-8")
    references = raw.split("\f")
    if references and not references[-1].strip():
        references.pop()
    if not any(t.strip() for t in references):
        raise RuntimeError("No PDF text layer: a labeled dataset or verified reference is required.")
    pages = []
    with pdfium.PdfDocument(str(pdf)) as document:
        count = min(limit or len(document), len(document))
        if len(references) != count:
            raise RuntimeError("PDF image and reference page counts differ.")
        for index in range(count):
            image_path = folder / f"page_{index + 1:03}.png"
            reference_path = folder / f"page_{index + 1:03}.txt"
            page = document[index]
            try:
                bitmap = page.render(scale=dpi / 72)
                try:
                    bitmap.to_pil().convert("RGB").save(image_path)
                finally:
                    bitmap.close()
            finally:
                page.close()
            reference_path.write_text(references[index], encoding="utf-8")
            pages.append({"page": index + 1, "image": image_path.name,
                          "reference": reference_path.name, "image_sha256": sha256(image_path),
                          "reference_sha256": sha256(reference_path)})
    manifest = {
        "pdf_sha256": digest, "dpi": dpi, "renderer": f"pypdfium2 {version('pypdfium2')}",
        "reference_status": "unverified PDF text layer; scores measure agreement, not verified accuracy",
        "page_selection": f"first {count} PDF pages" if limit else "all PDF pages",
        "normalization": "Unicode NFC, whitespace collapsed, Markdown heading prefixes removed; case/punctuation retained; no line exclusions",
        "pdf_text_tool": subprocess.run([executable, "-v"], capture_output=True, text=True).stderr.strip(),
        "pages": pages,
    }
    save_json(manifest_path, manifest)
    return folder, manifest


def load_engine(engine, threads, paddle_overrides=None):
    if engine == "easyocr":
        import easyocr
        import torch

        if version("easyocr") != "1.7.2":
            raise RuntimeError("This comparison pins easyocr==1.7.2")
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        model_dir = Path(__file__).resolve().parent / ".ocr-models/easyocr"
        reader = easyocr.Reader(["en"], gpu=False, model_storage_directory=str(model_dir),
            download_enabled=False, verbose=False, quantize=True)
        if str(reader.device) != "cpu":
            raise RuntimeError("EasyOCR must use CPU")
        files = [model_dir / name for name in ("craft_mlt_25k.pth", "english_g2.pth")]
        hashes = {p.name: sha256(p) for p in files}

        def recognize(path):
            return "\n".join(reader.readtext(str(path), detail=0, paragraph=False,
                decoder="greedy", batch_size=1, workers=0))

        return recognize, {"version": version("easyocr"), "torch": torch.__version__,
            "torchvision": version("torchvision"), "device": str(reader.device), "languages": ["en"],
            "detector": "CRAFT", "recognizer": "english_g2", "decoder": "greedy", "batch_size": 1,
            "workers": 0, "paragraph": False, "quantize_requested": True,
            "quantized_module_types": sorted({type(m).__module__ + "." + type(m).__name__
                for m in reader.recognizer.modules() if "quantized" in type(m).__module__}),
            "torch_intra_op_threads": torch.get_num_threads(), "torch_inter_op_threads": torch.get_num_interop_threads(),
            "model_hashes": hashes}
    if engine == "indra":
        import indra_ocr
        converter = indra_ocr.load(threads=threads, mode="ocr", furniture=True)

        def recognize(path):
            _, page = next(converter.convert(path))
            if page.source == "error":
                raise RuntimeError(page.note)
            return page.text

        models = Path(indra_ocr.model_dir())
        metadata = {"version": version("indra-ocr"), "mode": "ocr", "furniture": True, "output": "page.text (plain recognized lines)",
                    "model_hashes": {p.name: sha256(p) for p in models.rglob("*.onnx")}}
        return recognize, metadata
    if engine == "tesseract":
        executable = tesseract_path()

        def recognize(path):
            result = subprocess.run(
                [executable, str(path), "stdout", "-l", "eng", "--oem", "1", "--psm", "3"],
                check=True, capture_output=True
            )
            return result.stdout.decode("utf-8")

        langs = subprocess.run([executable, "--list-langs"], capture_output=True, text=True, check=True)
        if "eng" not in langs.stdout.split():
            raise RuntimeError("Tesseract English language data is missing.")
        traineddata = Path(executable).parent / "tessdata/eng.traineddata"
        metadata = {"version": subprocess.run([executable, "--version"], capture_output=True, text=True, check=True).stdout.strip(),
                    "language": "eng", "oem": 1, "psm": 3,
                    "language_data_sha256": sha256(traineddata) if traineddata.exists() else "not located",
                    "timing_note": "CLI process startup included per page; Python engines stay resident"}
        return recognize, metadata
    if engine == "paddleocr_tiny":
        paddle_overrides = {"text_detection_model_name": "PP-OCRv6_tiny_det",
            "text_recognition_model_name": "PP-OCRv6_tiny_rec", **(paddle_overrides or {})}
    elif engine != "paddleocr":
        raise ValueError(f"Unknown engine: {engine}")
    if version("paddleocr") != "3.7.0":
        raise RuntimeError("This comparison pins paddleocr==3.7.0; install the pinned version.")
    if version("paddlepaddle") != "3.2.2":
        raise RuntimeError("PaddlePaddle 3.3 has a reproduced CPU backend crash. Install paddlepaddle==3.2.2 before rerunning PaddleOCR.")
    from paddleocr import PaddleOCR
    settings = {"text_detection_model_name": "PP-OCRv6_medium_det",
                "text_recognition_model_name": "PP-OCRv6_medium_rec",
                "use_doc_orientation_classify": False, "use_doc_unwarping": False,
                "use_textline_orientation": False, "device": "cpu", "cpu_threads": threads,
                # Use Paddle 3.2.2: 3.3's oneDNN/PIR path crashes on this machine.
                "enable_mkldnn": True}
    overrides = dict(paddle_overrides or {})
    flush_denormals = overrides.pop("_onnx_flush_denormals", False)
    settings.update(overrides)
    if flush_denormals:
        import onnxruntime as ort
        original_options = ort.SessionOptions

        def cpu_options():
            options = original_options()
            options.add_session_config_entry("session.set_denormal_as_zero", "1")
            return options

        # PaddleX 3.7.2 does not expose ORT config entries through engine_config.
        ort.SessionOptions = cpu_options
        try:
            model = PaddleOCR(**settings)
        finally:
            ort.SessionOptions = original_options
    else:
        model = PaddleOCR(**settings)

    def recognize(path):
        results = list(model.predict(str(path)))
        if len(results) != 1:
            raise RuntimeError("Expected one result for one page image.")
        return "\n".join(results[0]["rec_texts"])

    pipeline = model.paddlex_pipeline._pipeline
    runtime, model_hashes = {}, {}
    for name, component in (("detector", pipeline.text_det_model), ("recognizer", pipeline.text_rec_model)):
        runtime[name] = dict(component.runner._config)
        if hasattr(component.runner, "session"):
            session = component.runner.session
            options = session.get_session_options()
            runtime[name].update(actual_providers=session.get_providers(),
                actual_intra_op_threads=options.intra_op_num_threads,
                actual_inter_op_threads=options.inter_op_num_threads)
            try:
                runtime[name]["denormal_as_zero"] = options.get_session_config_entry("session.set_denormal_as_zero")
            except RuntimeError:
                runtime[name]["denormal_as_zero"] = "0"
        model_hashes[name] = {p.name: sha256(p) for p in Path(component.runner.model_dir).glob("inference.*") if p.is_file()}
        if not model_hashes[name]:
            raise RuntimeError(f"Cannot record Paddle {name} model hashes")
    return recognize, {"version": version("paddleocr"), "paddlepaddle": version("paddlepaddle"),
                       "paddlex": version("paddlex"), "settings": settings,
                       "onnxruntime": version("onnxruntime") if settings.get("engine") == "onnxruntime" else None,
                       "effective_runtime": runtime, "model_hashes": model_hashes}


def configured_engine(engine, threads, paddle_backend="mkldnn", onnx_model_dir=None):
    if engine in ("paddleocr", "paddleocr_tiny") and paddle_backend in ("onnx", "onnx-ftz"):
        if onnx_model_dir is None:
            raise ValueError("ONNX backend requires a converted model directory")
        for name in ("detector", "recognizer"):
            if not (onnx_model_dir / name / "inference.onnx").is_file():
                raise ValueError(f"Missing converted ONNX {name}")
        return load_engine(engine, threads, {
            "_onnx_flush_denormals": paddle_backend == "onnx-ftz",
            "engine": "onnxruntime",
            "text_detection_model_dir": str(onnx_model_dir / "detector"),
            "text_recognition_model_dir": str(onnx_model_dir / "recognizer"),
            "engine_config": {"onnxruntime": {"providers": ["CPUExecutionProvider"],
                "intra_op_num_threads": threads, "inter_op_num_threads": 1}},
        })
    if engine in ("paddleocr", "paddleocr_tiny") and paddle_backend == "plain":
        return load_engine(engine, threads, {
            "enable_mkldnn": False,
            "engine_config": {"paddle_static": {"cpu_threads": threads, "run_mode": "paddle"}},
        })
    return load_engine(engine, threads)


def run_engine(engine, folder, manifest, threads, repeats, paddle_backend="mkldnn", onnx_model_dir=None):
    os.environ["OMP_THREAD_LIMIT"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    started = time.perf_counter()
    runner_hash = sha256(Path(__file__))
    recognize, metadata = configured_engine(engine, threads, paddle_backend, onnx_model_dir)
    initialization = time.perf_counter() - started
    # Warm up on the first page; excluded from reported throughput for every engine.
    warmup = time.perf_counter()
    recognize(folder / manifest["pages"][0]["image"])
    warmup = time.perf_counter() - warmup
    run = folder / engine / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True)
    pages, passes, errors, pass_details = [], [], [], []
    for trial in range(repeats):
        trial_seconds = 0
        trial_pages = []
        for page in manifest["pages"]:
            started = time.perf_counter()
            try:
                prediction = recognize(folder / page["image"])
                failure = None
            except Exception as error:
                prediction, failure = "", str(error)
                errors.append({"trial": trial + 1, "page": page["page"], "error": failure})
            seconds = time.perf_counter() - started
            trial_seconds += seconds
            reference = (folder / page["reference"]).read_text(encoding="utf-8")
            normalized_prediction = normalize(prediction, [])
            score = compare(normalize(reference, []), normalized_prediction)
            trial_pages.append({"page": page["page"], "seconds": seconds, "error": failure,
                "normalized_text_sha256": hashlib.sha256(normalized_prediction.encode("utf-8")).hexdigest(),
                "score": {k: v for k, v in score.items() if k != "changes"}})
            if trial == 0:
                (run / f"page_{page['page']:03}.txt").write_text(prediction, encoding="utf-8")
                pages.append({"page": page["page"], "seconds": seconds, "error": failure,
                              "score": score})
            print(f"{engine}: pass {trial + 1}/{repeats}, page {page['page']}/{len(manifest['pages'])}, {seconds:.2f}s", flush=True)
        passes.append(trial_seconds)
        pass_details.append({"trial": trial + 1, "seconds": trial_seconds,
                             "overall": aggregate(trial_pages, "score"), "pages": trial_pages})
    result = {"engine": engine, "metadata": metadata, "dataset_sha256": sha256(folder / "dataset.json"),
              "unit": manifest.get("unit", "pages"),
              "hardware": {"platform": platform.platform(), "processor": os.getenv("PROCESSOR_IDENTIFIER", platform.processor()),
                           "logical_cpus": os.cpu_count(), "device": "CPU", "threads": threads},
              "python": sys.version, "runner_sha256": runner_hash,
              "initialization_seconds": initialization, "warmup_seconds": warmup,
              "pass_seconds": passes, "median_pass_seconds": statistics.median(passes),
              "pages_per_minute": len(pages) * 60 / statistics.median(passes),
              "timing_scope": "serial image-to-text including file decode, excludes rendering, writes and scoring",
              "scoring_scope": "numerical scores and normalized text hashes for all passes; raw predictions and differences for first pass; failures scored as empty predictions",
              "pass_details": pass_details,
              "errors": errors, "overall": aggregate(pages, "score"), "pages": pages}
    save_json(run / "results.json", result)
    print(f"Saved {run / 'results.json'}", flush=True)


def summarize(folder, manifest):
    unit = manifest.get("unit", "pages")
    engines = dataset_engines(manifest)
    results = []
    for engine in engines:
        files = sorted((folder / engine).glob("*/results.json"))
        if files:
            result = json.loads(files[-1].read_text(encoding="utf-8"))
            if result["dataset_sha256"] != sha256(folder / "dataset.json"):
                raise RuntimeError("Results belong to a different dataset manifest.")
            results.append(result)
    lines = ["# " + " vs ".join(engines) + ": image-only CPU comparison", "",
             "Draft results. " + manifest["reference_status"] + ".", "",
             f"| Engine | WER | CER | {unit.capitalize()}/min | Median pass (s) | Passes | Failures |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['engine']} | {percent(r['overall']['wer'])} | {percent(r['overall']['cer'])} | {r['pages_per_minute']:.2f} | {r['median_pass_seconds']:.2f} | {len(r['pass_seconds'])} | {len(r['errors'])} |")
    missing = sorted(set(engines) - {r["engine"] for r in results})
    lines += ["", f"Missing engines: {', '.join(missing) or 'none'}.", "",
              f"Dataset: {len(manifest['pages'])} {unit}; resolution: {manifest.get('dpi', 'original pixels')}; {manifest.get('page_selection', manifest.get('scope', 'all items'))}.",
              "All engines received identical RGB PNGs, never the source PDF. Indra was forced into OCR mode with headers/footers retained.",
              "One untimed warmup page per engine. Reported speed is serial image-to-text; downloads, initialization, PDF rendering, scoring and output writes are excluded.",
              "Tesseract uses a process per page; its process startup is included, unlike resident Python models. This is application throughput, not an isolated neural inference comparison.",
              "PaddleOCR model tier, weights and backend settings are recorded in results.json. PaddlePaddle is pinned to 3.2.2 because the 3.3.0 oneDNN/PIR path crashed during warmup on this Windows machine.",
              "WER/CER use summed Levenshtein errors divided by summed reference lengths. Case and punctuation count. Only NFC, whitespace and Markdown heading prefixes are normalized. No header/footer exclusions.",
              "The headline quality scores use the first pass; newer runs also retain scores and normalized text hashes for every repeat. Failures remain in the denominator and are scored as empty output.",
              "", ("Custom annotated text-region subset; gold crop boundaries are supplied. This is not an official OmniDocBench score and does not evaluate full-page detection or reading order. Annotation errors and possible training overlap remain limitations."
                     if manifest.get("kind") == "public_annotated_text_regions" else
                     "This is one English book and an unverified text-layer reference. It does not establish a universal OCR winner or independently verified accuracy. A multi-document labeled benchmark is needed for broader claims."),
              "The earlier 0.17% auto-mode result is excluded: Indra could read the same PDF text layer used for reference.",
              "", "Detailed versions, model identifiers, hashes, settings, page differences and timing are in each engine's results.json.",
              "No book text or page images should be uploaded with the article without appropriate rights."]
    (folder / "medium-draft.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(folder / "comparison.json", {"dataset": manifest, "missing_engines": missing, "results": results})
    from benchmark_stats import build_statistics
    build_statistics(folder)
    rows = []
    for r in results:
        for page in r["pages"]:
            for change in page["score"]["changes"]:
                rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in
                    (r["engine"], page["page"], change["type"], change["reference"], change["prediction"], change["context"])) + "</tr>")
    (folder / "comparison.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>OCR comparison</title><style>body{font:16px system-ui;margin:30px}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}input{padding:10px;width:70%}</style>'
        + '<p><a href="publication-stats.md">Detailed publication statistics</a> · <a href="page-stats.csv">Per-page CSV</a> · <a href="publication-stats.json">Statistics JSON</a></p>'
        + '<h1>Image-only OCR comparison</h1><pre>' + html.escape("\n".join(lines)) + '</pre>'
        + '<h2>Word changes</h2><input id="q" placeholder="Filter engine, word or page" aria-label="Filter changes"><table><thead><tr><th>Engine</th><th>Page</th><th>Type</th><th>Reference</th><th>Prediction</th><th>Context</th></tr></thead><tbody>'
        + ''.join(rows) + '</tbody></table><script>document.querySelector("#q").oninput=e=>document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()))</script>', encoding="utf-8")
    print("\n".join(lines[:11]))
    print(f"Report: {(folder / 'comparison.html').resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=(*ENGINES, *ADDITIONAL_ENGINES, "all", "summary"), default="all")
    parser.add_argument("--pdf", type=Path, default=Path("Rich_Dad_Poor_Dad.pdf"))
    parser.add_argument("--dataset", type=Path, help="Prepared image/reference dataset with dataset.json; no PDF extraction")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--paddle-backend", choices=("mkldnn", "plain", "onnx", "onnx-ftz"), default="mkldnn")
    parser.add_argument("--paddle-model-dir", type=Path)
    args = parser.parse_args()
    if args.dpi <= 0 or args.limit < 0 or args.threads <= 0 or args.repeats <= 0:
        parser.error("DPI, threads and repeats must be positive; limit must be nonnegative.")
    if args.dataset:
        folder, manifest = args.dataset, load_dataset(args.dataset)
    else:
        folder, manifest = prepare(args.pdf, args.dpi, args.limit)
    if args.engine == "all":
        failures = []
        for engine in dataset_engines(manifest):
            log = folder / f"{engine}.log"
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--engine", engine,
                       "--pdf", str(args.pdf), "--dpi", str(args.dpi), "--limit", str(args.limit),
                       "--threads", str(args.threads), "--repeats", str(args.repeats),
                       "--paddle-backend", args.paddle_backend]
            if args.dataset:
                command += ["--dataset", str(args.dataset)]
            if args.paddle_model_dir:
                command += ["--paddle-model-dir", str(args.paddle_model_dir)]
            print(f"Running {engine}; progress log: {log}", flush=True)
            with log.open("w", encoding="utf-8") as output:
                status = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT).returncode
            if status:
                failures.append(f"{engine} (see {log})")
        summarize(folder, manifest)
        if failures:
            parser.exit(1, "Incomplete comparison: " + ", ".join(failures) + "\n")
    elif args.engine == "summary":
        summarize(folder, manifest)
    else:
        run_engine(args.engine, folder, manifest, args.threads, args.repeats, args.paddle_backend, args.paddle_model_dir)


if __name__ == "__main__":
    main()
