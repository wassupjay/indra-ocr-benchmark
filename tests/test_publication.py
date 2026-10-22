import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from benchmark import ENGINES, save_json
from evaluate import aggregate, compare, sha256
import hashlib
from finish_publication import completed_run
from publication_report import audit_dataset, compare_saved_outputs


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path("eval/test-runs") / uuid4().hex
        self.folder.mkdir(parents=True)
        (self.folder / "page_001.png").write_bytes(b"hashed fixture, not decoded")
        (self.folder / "page_001.txt").write_text("one two", encoding="utf-8")
        save_json(self.folder / "dataset.json", {"reference_status": "test", "pages": [{"page": 1,
            "image": "page_001.png", "reference": "page_001.txt",
            "image_sha256": sha256(self.folder / "page_001.png"), "reference_sha256": sha256(self.folder / "page_001.txt")}]})
        pages = [{"page": 1, "seconds": 1, "score": compare("one two", "one") }]
        for engine in ENGINES:
            run = self.folder / engine / "run"
            run.mkdir(parents=True)
            (run / "page_001.txt").write_text("one", encoding="utf-8")
            save_json(run / "results.json", {"pages": pages, "overall": aggregate(pages, "score"),
                "dataset_sha256": sha256(self.folder / "dataset.json"), "pass_seconds": [1, 2, 3],
                "median_pass_seconds": 2, "pages_per_minute": 30, "errors": [],
                "runner_sha256": sha256(Path("benchmark.py")), "hardware": {"threads": 4},
                "metadata": {"settings": {"enable_mkldnn": False}}})

    def tearDown(self):
        assert self.folder.resolve().parent == Path("eval/test-runs").resolve()
        shutil.rmtree(self.folder)

    def test_audit_recomputes_saved_predictions(self):
        with patch("publication_report.summarize"):
            result = audit_dataset(self.folder)
        self.assertEqual(result["engines"]["indra"]["overall"]["word_errors"], 1)
        self.assertEqual(result["engines"]["indra"]["pages_recomputed"], 1)

    def test_audit_rejects_changed_prediction(self):
        (self.folder / "indra/run/page_001.txt").write_text("one two", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "prediction/score mismatch"):
            audit_dataset(self.folder)

    def test_saved_output_comparison_checks_text_not_just_edit_totals(self):
        with patch("publication_report.summarize"):
            evidence = audit_dataset(self.folder)
        earlier_dir = self.folder / "earlier"
        earlier_dir.mkdir()
        (earlier_dir / "page_001.txt").write_text("one  ", encoding="utf-8")
        earlier = {"engines": {engine: {**item, "result_file": str(earlier_dir / "results.json")}
                   for engine, item in evidence["engines"].items()}}
        self.assertTrue(compare_saved_outputs(evidence, earlier)["indra"]["all_normalized_outputs_match"])
        (earlier_dir / "page_001.txt").write_text("two", encoding="utf-8")
        self.assertEqual(compare_saved_outputs(evidence, earlier)["indra"]["changed_prediction_files"], ["page_001.txt"])

    def test_audit_rejects_wrong_speed(self):
        path = self.folder / "indra/run/results.json"
        result = json.loads(path.read_text())
        result["pages_per_minute"] = 999
        save_json(path, result)
        with self.assertRaisesRegex(ValueError, "throughput"):
            audit_dataset(self.folder)

    def test_audit_rejects_different_score_for_identical_output(self):
        path = self.folder / "indra/run/results.json"
        result = json.loads(path.read_text())
        score = {k: v for k, v in result["pages"][0]["score"].items() if k != "changes"}
        score["wer"] = 0.75
        result["pass_details"] = [{"pages": [{"page": 1, "seconds": s, "score": score,
            "normalized_text_sha256": hashlib.sha256(b"one").hexdigest()}],
            "overall": aggregate([{"score": score}], "score")} for s in (1, 2, 3)]
        save_json(path, result)
        with self.assertRaisesRegex(ValueError, "different score"):
            audit_dataset(self.folder)

    def test_onnx_resume_requires_matching_weights_provider_and_ftz(self):
        model_dir = self.folder / "models"
        metadata = {"settings": {"engine": "onnxruntime"}, "effective_runtime": {}, "model_hashes": {}}
        for name in ("detector", "recognizer"):
            component = model_dir / name
            component.mkdir(parents=True)
            (component / "inference.onnx").write_bytes(b"model")
            metadata["model_hashes"][name] = {"inference.onnx": sha256(component / "inference.onnx")}
            metadata["effective_runtime"][name] = {"actual_providers": ["CPUExecutionProvider"],
                "actual_intra_op_threads": 4, "actual_inter_op_threads": 1, "denormal_as_zero": "1"}
        path = self.folder / "paddleocr/run/results.json"
        result = json.loads(path.read_text())
        result["metadata"] = metadata
        save_json(path, result)
        self.assertIsNotNone(completed_run(self.folder, "paddleocr", 4, 3, "onnx-ftz", model_dir))
        self.assertIsNone(completed_run(self.folder, "paddleocr", 4, 3))
        metadata["effective_runtime"]["recognizer"]["denormal_as_zero"] = "0"
        save_json(path, result)
        self.assertIsNone(completed_run(self.folder, "paddleocr", 4, 3, "onnx-ftz", model_dir))

    def test_resume_accepts_only_matching_successful_plain_backend(self):
        self.assertIsNotNone(completed_run(self.folder, "paddleocr", 4, 3))
        self.assertIsNone(completed_run(self.folder, "paddleocr", 2, 3))
        path = self.folder / "paddleocr/run/results.json"
        result = json.loads(path.read_text())
        result["metadata"]["settings"]["enable_mkldnn"] = True
        save_json(path, result)
        self.assertIsNone(completed_run(self.folder, "paddleocr", 4, 3))

    def test_resume_does_not_hide_a_newer_failed_run(self):
        older = self.folder / "paddleocr/run/results.json"
        newer = self.folder / "paddleocr/z-new"
        newer.mkdir()
        result = json.loads(older.read_text())
        result["errors"] = [{"page": 1}]
        save_json(newer / "results.json", result)
        self.assertIsNone(completed_run(self.folder, "paddleocr", 4, 3))


if __name__ == "__main__":
    unittest.main()
