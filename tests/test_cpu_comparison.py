import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import benchmark
from benchmark import dataset_engines, save_json
from evaluate import sha256
from publication_report import audit_dataset
import run_cpu_comparison as cpu


class CpuComparisonTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path("eval/test-runs") / uuid4().hex
        self.folder.mkdir(parents=True)

    def tearDown(self):
        assert self.folder.resolve().parent == Path("eval/test-runs").resolve()
        shutil.rmtree(self.folder)

    def fixture(self):
        for name, data in (("page_001.png", b"fixture bytes, never decoded"), ("page_001.txt", b"one two")):
            (self.folder / name).write_bytes(data)
        manifest = {"reference_status": "fixture", "engines": list(cpu.ENGINES),
            "pages": [{"page": 1, "image": "page_001.png", "reference": "page_001.txt",
                "image_sha256": sha256(self.folder / "page_001.png"),
                "reference_sha256": sha256(self.folder / "page_001.txt")}]}
        save_json(self.folder / "dataset.json", manifest)
        return manifest

    def test_engine_selection_preserves_history_and_rejects_invalid_lists(self):
        self.assertEqual(dataset_engines({}), ("indra", "tesseract", "paddleocr"))
        self.assertEqual(dataset_engines({"engines": list(cpu.ENGINES)}), cpu.ENGINES)
        for names in ([], ["easyocr", "easyocr"], ["../escape"], "easyocr"):
            with self.assertRaises(ValueError):
                dataset_engines({"engines": names})

    def test_four_engine_scores_are_audited_and_resume_checks_settings_and_resources(self):
        manifest = self.fixture()
        metadata = {"version": "fixture", "model_hashes": {"model": "original"}}
        with patch.object(benchmark, "configured_engine", return_value=(lambda path: "one", metadata)):
            for engine in cpu.ENGINES:
                benchmark.run_engine(engine, self.folder, manifest, 4, 3)
                save_json(self.folder / f"{engine}-resources.json", {"exit_code": 0, "timed_out": False,
                    "sampled_peak_tree_rss_mib": 10})
        evidence = audit_dataset(self.folder)
        self.assertEqual(set(evidence["engines"]), set(cpu.ENGINES))
        self.assertTrue(all(e["overall"]["word_errors"] == 1 for e in evidence["engines"].values()))
        self.assertIsNotNone(cpu.matching_result(self.folder, "easyocr", metadata))
        self.assertIsNone(cpu.matching_result(self.folder, "easyocr", {"version": "different"}))
        (self.folder / "publication").mkdir()
        with patch.object(cpu, "ROOT", self.folder):
            cpu.write_report(self.folder)
        self.assertTrue((self.folder / "publication/cpu-four-results.png").is_file())
        report = json.loads((self.folder / "publication/cpu-four-comparison.json").read_text())
        self.assertEqual(set(report["statistics"]["models"]), set(cpu.ENGINES))
        (self.folder / "easyocr-resources.json").unlink()
        self.assertIsNone(cpu.matching_result(self.folder, "easyocr", metadata))

    def test_easyocr_uses_cpu_cached_weights_and_plain_text_output(self):
        reader = SimpleNamespace(device="cpu", readtext=Mock(return_value=["first", "second"]),
                                 recognizer=SimpleNamespace(modules=lambda: []))
        easy = SimpleNamespace(Reader=Mock(return_value=reader))
        torch = SimpleNamespace(__version__="2.8.0+cpu", set_num_threads=Mock(), set_num_interop_threads=Mock(),
            get_num_threads=lambda: 4, get_num_interop_threads=lambda: 1)
        with patch.dict("sys.modules", {"easyocr": easy, "torch": torch}), \
             patch.object(benchmark, "version", side_effect=lambda n: {"easyocr": "1.7.2", "torchvision": "0.23.0+cpu"}[n]), \
             patch.object(benchmark, "sha256", return_value="hash"):
            recognize, metadata = benchmark.load_engine("easyocr", 4)
        self.assertEqual(recognize(Path("page.png")), "first\nsecond")
        self.assertFalse(easy.Reader.call_args.kwargs["gpu"])
        self.assertFalse(easy.Reader.call_args.kwargs["download_enabled"])
        self.assertEqual(reader.readtext.call_args.kwargs["workers"], 0)
        self.assertEqual(metadata["torch_intra_op_threads"], 4)
        torch.set_num_interop_threads.assert_called_once_with(1)

    def test_tiny_adapter_selects_tiny_instead_of_medium(self):
        for file in ("inference.onnx", "inference.yml"):
            (self.folder / file).write_text("fixture", encoding="utf-8")
        component = SimpleNamespace(runner=SimpleNamespace(_config={}, model_dir=self.folder))
        model = SimpleNamespace(paddlex_pipeline=SimpleNamespace(_pipeline=SimpleNamespace(
            text_det_model=component, text_rec_model=component)))
        ctor = Mock(return_value=model)
        with patch.dict("sys.modules", {"paddleocr": SimpleNamespace(PaddleOCR=ctor)}), \
             patch.object(benchmark, "version", side_effect=lambda n: {"paddleocr": "3.7.0", "paddlepaddle": "3.2.2", "paddlex": "3.7.2"}[n]):
            benchmark.load_engine("paddleocr_tiny", 4)
        self.assertEqual(ctor.call_args.kwargs["text_detection_model_name"], "PP-OCRv6_tiny_det")
        self.assertEqual(ctor.call_args.kwargs["text_recognition_model_name"], "PP-OCRv6_tiny_rec")

    def test_cached_tiny_provenance_rejects_tampering_and_missing_hashes(self):
        provenance = {"models": {}}
        for component in ("detector", "recognizer"):
            (self.folder / component).mkdir()
            for file in ("inference.onnx", "inference.yml"):
                (self.folder / component / file).write_text("model", encoding="utf-8")
            provenance["models"][component] = {"files": {name: sha256(self.folder / component / name)
                                                        for name in ("inference.onnx", "inference.yml")}}
        save_json(self.folder / "provenance.json", provenance)
        with patch.object(cpu, "MODELS", self.folder):
            cpu.prepare_tiny()
            (self.folder / "detector/inference.onnx").write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                cpu.prepare_tiny()
            provenance["models"]["detector"]["files"] = {}
            save_json(self.folder / "provenance.json", provenance)
            with self.assertRaisesRegex(ValueError, "both model and config"):
                cpu.prepare_tiny()

    def test_setup_network_failure_precedes_environment_changes(self):
        with patch.object(cpu, "fetch_json", side_effect=OSError("offline")), patch.object(cpu.subprocess, "run") as run:
            with self.assertRaises(OSError):
                cpu.setup()
        run.assert_not_called()

    def test_download_rejects_an_invalid_model_revision(self):
        with patch.object(cpu, "MODELS", self.folder), patch.object(cpu, "fetch_json", return_value={"sha": "../escape"}):
            with self.assertRaisesRegex(ValueError, "Invalid Hugging Face"):
                cpu.prepare_tiny()
        self.assertFalse((self.folder / "detector").exists())

    def test_missing_assets_detects_an_interpreter_with_broken_dependencies(self):
        python = self.folder / "python.exe"
        python.write_bytes(b"fixture")
        with patch.object(cpu, "ROOT", self.folder), patch.object(cpu, "EASY_PYTHON", python), \
             patch.object(cpu, "MODELS", self.folder / "models"), \
             patch.object(cpu.subprocess, "run", return_value=SimpleNamespace(returncode=1)):
            self.assertIn("EasyOCR/PyTorch CPU dependencies", cpu.missing_assets())


if __name__ == "__main__":
    unittest.main()
