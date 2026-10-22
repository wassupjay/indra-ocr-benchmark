import json
from pathlib import Path
import shutil
import sys
import unittest
from uuid import uuid4

from benchmark_stats import build_statistics, percentile, rate_samples
from evaluate import aggregate, compare
from profile_ocr import monitor


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path("eval/test-runs") / uuid4().hex
        self.folder.mkdir(parents=True)

    def tearDown(self):
        assert self.folder.resolve().parent == Path("eval/test-runs").resolve()
        shutil.rmtree(self.folder)

    def test_percentile_interpolation(self):
        self.assertEqual(percentile([0, 10], .95), 9.5)
        self.assertEqual(percentile([3], .5), 3)

    def test_bootstrap_uses_weighted_reference_denominators(self):
        self.assertEqual(rate_samples([1, 0], [2, 8], [[0, 1], [0, 0]]), [.1, .5])

    def test_paired_identical_models_have_zero_difference(self):
        pages = [{"page": 1, "seconds": .1, "score": compare("one two", "one")},
                 {"page": 2, "seconds": .2, "score": compare("three", "three")}]
        base = {"pages": pages, "overall": aggregate(pages, "score"), "errors": [],
                "pass_seconds": [1, 2, 3], "median_pass_seconds": 2,
                "pages_per_minute": 60, "initialization_seconds": .1,
                "metadata": {}, "hardware": {}}
        source = {"dataset": {"reference_status": "test"}, "missing_engines": [],
                  "results": [{**base, "engine": "left"}, {**base, "engine": "right"}]}
        (self.folder / "comparison.json").write_text(json.dumps(source))
        result = build_statistics(self.folder, replicates=100)
        self.assertEqual(result["paired_comparisons"][0]["wer_difference_95"], [0, 0])
        self.assertEqual(result["models"]["left"]["word_edits"], 1)
        self.assertTrue((self.folder / "page-stats.csv").is_file())

    def test_profiler_stops_only_its_own_timed_out_worker(self):
        result = monitor([sys.executable, "-c", "import time; time.sleep(5)"], .2, self.folder / "log.txt")
        self.assertTrue(result["timed_out"])
        self.assertGreater(result["sampled_peak_tree_rss_mib"], 0)


if __name__ == "__main__":
    unittest.main()
