from copy import deepcopy
from pathlib import Path
import shutil
import unittest
from uuid import uuid4
import zipfile

from compare_onnx import validate_comparison
from run_deployment_comparison import select_pages
from convert_onnx import unpack_wheel
from evaluate import sha256


class OnnxComparisonTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"source_dataset_sha256": "input", "pages": [{"page": 1}]}
        page = {"page": 1, "normalized_text_sha256": "same"}
        trial = {"pages": [page], "overall": {"word_errors": 0, "character_errors": 0}}
        self.baseline = {"source_dataset_sha256": "input", "raw_pass_details": [deepcopy(trial)],
            "metadata": {"model_hashes": {k: {"weights": "original"} for k in ("detector", "recognizer")}}}
        self.conversion = {"models": {k: {"source_sha256": {"weights": "original"}} for k in ("detector", "recognizer")}}
        self.result = {"errors": [], "pass_details": [deepcopy(trial) for _ in range(3)],
            "metadata": {"effective_runtime": {k: {"actual_providers": ["CPUExecutionProvider"], "actual_intra_op_threads": 4}
                         for k in ("detector", "recognizer")}}}

    def test_matching_model_inputs_and_outputs_pass(self):
        self.assertTrue(all(validate_comparison(self.result, self.baseline, self.conversion, self.manifest).values()))

    def test_deployment_selection_is_fixed_and_ignores_scores(self):
        pages = [{"page": i, "wer": i / 114} for i in range(1, 115)]
        selected = select_pages(pages)
        self.assertEqual(selected, select_pages([{**p, "wer": 0} for p in pages]))
        self.assertEqual([i + 1 for i in selected], [9, 15, 26, 43, 59, 65, 70, 100, 101, 102, 110, 111])

    def test_changed_source_weights_are_rejected(self):
        self.conversion["models"]["detector"]["source_sha256"]["weights"] = "different"
        with self.assertRaisesRegex(ValueError, "source weights"):
            validate_comparison(self.result, self.baseline, self.conversion, self.manifest)

    def test_wrong_provider_is_rejected(self):
        self.result["metadata"]["effective_runtime"]["recognizer"]["actual_providers"] = ["CUDAExecutionProvider"]
        with self.assertRaisesRegex(ValueError, "CPU provider"):
            validate_comparison(self.result, self.baseline, self.conversion, self.manifest)

    def test_quality_changes_are_reported(self):
        self.result["pass_details"][1]["overall"]["word_errors"] = 1
        self.result["pass_details"][1]["pages"][0]["normalized_text_sha256"] = "changed"
        checks = validate_comparison(self.result, self.baseline, self.conversion, self.manifest)
        self.assertFalse(checks["all_passes_zero_edits"])
        self.assertFalse(checks["normalized_outputs_match_baseline"])

    def test_wheel_unpack_rejects_path_escape_and_wrong_checksum(self):
        folder = Path("eval/test-runs") / uuid4().hex
        folder.mkdir(parents=True)
        try:
            wheel = folder / "bad.whl"
            with zipfile.ZipFile(wheel, "w") as package:
                package.writestr("../outside.txt", "reject")
            with self.assertRaisesRegex(ValueError, "checksum"):
                unpack_wheel(wheel, folder / "target", "wrong")
            with self.assertRaisesRegex(ValueError, "outside"):
                unpack_wheel(wheel, folder / "target", sha256(wheel))
            self.assertFalse((folder / "outside.txt").exists())
        finally:
            assert folder.resolve().parent == Path("eval/test-runs").resolve()
            shutil.rmtree(folder)


if __name__ == "__main__":
    unittest.main()
