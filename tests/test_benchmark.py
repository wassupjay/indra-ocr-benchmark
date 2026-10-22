import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

import benchmark


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path("eval/test-runs") / uuid4().hex
        self.folder.mkdir(parents=True)

    def tearDown(self):
        root = Path("eval/test-runs").resolve()
        assert self.folder.resolve().parent == root
        shutil.rmtree(self.folder)

    def test_failed_page_is_scored_instead_of_dropped(self):
        folder = self.folder
        (folder / "dataset.json").write_text("{}")
        (folder / "page_001.txt").write_text("one two three")
        manifest = {"pages": [{"page": 1, "image": "page_001.png", "reference": "page_001.txt"}]}
        calls = []

        def recognize(path):
            calls.append(path)
            if len(calls) > 1:
                raise RuntimeError("test failure")
            return "warmup"

        with patch.object(benchmark, "load_engine", return_value=(recognize, {})):
            benchmark.run_engine("indra", folder, manifest, 1, 1)
        result_path = next((folder / "indra").glob("*/results.json"))
        result = json.loads(result_path.read_text())
        self.assertEqual(result["overall"]["wer"], 1)
        self.assertEqual(result["overall"]["reference_words"], 3)
        self.assertEqual(len(result["errors"]), 1)
        self.assertTrue(all(path.suffix == ".png" for path in calls))

    def test_changed_dataset_is_rejected(self):
        folder = self.folder
        (folder / "dataset.json").write_text("{}")
        result_folder = folder / "indra" / "run"
        result_folder.mkdir(parents=True)
        (result_folder / "results.json").write_text('{"dataset_sha256":"wrong"}')
        with self.assertRaisesRegex(RuntimeError, "different dataset"):
            benchmark.summarize(folder, {})

    def test_plain_paddle_backend_is_passed_explicitly(self):
        folder = self.folder
        (folder / "dataset.json").write_text("{}")
        (folder / "page_001.txt").write_text("one")
        manifest = {"pages": [{"page": 1, "image": "page_001.png", "reference": "page_001.txt"}]}
        with patch.object(benchmark, "load_engine", return_value=(lambda path: "one", {})) as load:
            benchmark.run_engine("paddleocr", folder, manifest, 4, 3, "plain")
        settings = load.call_args.args[2]
        self.assertFalse(settings["enable_mkldnn"])
        self.assertEqual(settings["engine_config"]["paddle_static"]["run_mode"], "paddle")
        self.assertEqual(settings["engine_config"]["paddle_static"]["cpu_threads"], 4)

    def test_later_pass_quality_changes_are_recorded(self):
        folder = self.folder
        (folder / "dataset.json").write_text("{}")
        (folder / "page_001.txt").write_text("one")
        manifest = {"pages": [{"page": 1, "image": "page_001.png", "reference": "page_001.txt"}]}
        outputs = iter(["warmup", "one", "two", "one"])
        with patch.object(benchmark, "load_engine", return_value=(lambda path: next(outputs), {})):
            benchmark.run_engine("indra", folder, manifest, 4, 3)
        result = json.loads(next((folder / "indra").glob("*/results.json")).read_text())
        self.assertEqual([p["overall"]["word_errors"] for p in result["pass_details"]], [0, 1, 0])
        hashes = [p["pages"][0]["normalized_text_sha256"] for p in result["pass_details"]]
        self.assertEqual(hashes[0], hashes[2])
        self.assertNotEqual(hashes[0], hashes[1])

    def test_onnx_backend_requests_cpu_threads_and_explicit_float_setting(self):
        folder = self.folder
        (folder / "dataset.json").write_text("{}")
        (folder / "page_001.txt").write_text("one")
        for name in ("detector", "recognizer"):
            (folder / name).mkdir()
            (folder / name / "inference.onnx").write_bytes(b"mock model")
        manifest = {"pages": [{"page": 1, "image": "page_001.png", "reference": "page_001.txt"}]}
        with patch.object(benchmark, "load_engine", return_value=(lambda path: "one", {})) as load:
            benchmark.run_engine("paddleocr", folder, manifest, 4, 3, "onnx-ftz", folder)
        settings = load.call_args.args[2]
        self.assertEqual(settings["engine"], "onnxruntime")
        self.assertTrue(settings["_onnx_flush_denormals"])
        self.assertEqual(settings["engine_config"]["onnxruntime"]["intra_op_num_threads"], 4)
        self.assertEqual(settings["engine_config"]["onnxruntime"]["providers"], ["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
