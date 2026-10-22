from copy import deepcopy
import unittest

from verify_paddle import assess


class PaddleVerificationTests(unittest.TestCase):
    def setUp(self):
        self.pages = [{"page": i} for i in (1, 2, 3)]
        trial = {"overall": {"word_errors": 0, "character_errors": 0},
                 "pages": [{"page": i, "normalized_text_sha256": str(i) * 64} for i in (1, 2, 3)]}
        self.result = {"errors": [], "pass_seconds": [1, 1.1, 1.05],
            "pass_details": [deepcopy(trial) for _ in range(3)],
            "metadata": {"effective_runtime": {k: {"run_mode": "paddle", "cpu_threads": 4} for k in ("detector", "recognizer")},
                         "model_hashes": {k: {"inference.pdiparams": "a" * 64} for k in ("detector", "recognizer")}}}
        self.resources = {"timed_out": False, "exit_code": 0}

    def test_complete_consistent_configuration_passes(self):
        self.assertTrue(assess(self.result, self.resources, self.pages)["configuration_verified_on_control"])

    def test_changed_output_in_later_pass_fails(self):
        self.result["pass_details"][1]["pages"][0]["normalized_text_sha256"] = "changed"
        self.assertFalse(assess(self.result, self.resources, self.pages)["configuration_verified_on_control"])

    def test_later_accuracy_regression_fails(self):
        self.result["pass_details"][2]["overall"]["character_errors"] = 1
        self.assertFalse(assess(self.result, self.resources, self.pages)["configuration_verified_on_control"])

    def test_timeout_or_wrong_backend_fails(self):
        self.resources["timed_out"] = True
        self.assertFalse(assess(self.result, self.resources, self.pages)["configuration_verified_on_control"])
        self.resources["timed_out"] = False
        self.result["metadata"]["effective_runtime"]["recognizer"]["run_mode"] = "mkldnn"
        self.assertFalse(assess(self.result, self.resources, self.pages)["configuration_verified_on_control"])


if __name__ == "__main__":
    unittest.main()
