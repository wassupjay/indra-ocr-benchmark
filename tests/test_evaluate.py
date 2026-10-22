import unittest

from evaluate import aggregate, compare, normalize


class EvaluationTests(unittest.TestCase):
    def test_edit_types_and_reference_denominator(self):
        for expected, actual, kind in [
            ("one two three", "one four three", "Changed"),
            ("one two three", "one three", "Missing"),
            ("one two three", "one extra two three", "Added"),
        ]:
            result = compare(expected, actual)
            self.assertAlmostEqual(result["wer"], 1 / 3)
            self.assertEqual(result["changes"][0]["type"], kind)

    def test_normalization_preserves_meaningful_errors(self):
        result = normalize("## Title\nHeader\nCafe\u0301  10.0\n", ["Header"])
        self.assertEqual(result, "Title Café 10.0")
        self.assertNotEqual(result, normalize("title Café 100", []))

    def test_empty_reference_and_micro_average(self):
        self.assertIsNone(compare("", "extra")["wer"])
        pages = [{"score": compare("a b c", "a b")}, {"score": compare("d", "d")}]
        self.assertEqual(aggregate(pages, "score")["wer"], 0.25)

    def test_document_header_footer_rules(self):
        import json
        from pathlib import Path
        rules = json.loads(Path("eval/config.json").read_text())["exclude_lines"]
        text = "Rich Dad Poor Dad     Robert T. Kiyosaki\nReal text\nPage 5/114    http://motsach.info\n"
        self.assertEqual(normalize(text, rules), "Real text")


if __name__ == "__main__":
    unittest.main()
