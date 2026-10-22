import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from benchmark import load_dataset
from control_benchmark import TEXT, wrap
from prepare_public import select_regions
from validate_control import validate
from evaluate import sha256
import profile_ocr


class ControlTests(unittest.TestCase):
    def test_profile_forwards_backend_and_thread_settings_to_worker(self):
        dataset = Path("eval/test-runs") / uuid4().hex
        dataset.mkdir(parents=True)
        try:
            self.check_profile_forwarding(dataset)
        finally:
            assert dataset.resolve().parent == Path("eval/test-runs").resolve()
            shutil.rmtree(dataset)

    def check_profile_forwarding(self, dataset):
        Image.new("RGB", (5, 5), "white").save(dataset / "page_003.png")
        argv = ["profile_ocr.py", "--dataset", str(dataset), "--page", "3",
                "--engines", "paddleocr", "--threads", "1", "--paddle-no-mkldnn", "--skip-warmup"]
        with patch("sys.argv", argv), patch.object(Path, "mkdir"), \
             patch.object(profile_ocr, "save_json"), patch.object(profile_ocr, "write_profile_report"), \
             patch.object(profile_ocr, "monitor", return_value={"wall_seconds": 0, "sampled_peak_tree_rss_mib": 0, "timed_out": False}) as monitor:
            profile_ocr.main()
        command = monitor.call_args.args[0]
        self.assertEqual(command[command.index("--threads") + 1], "1")
        self.assertIn("--paddle-no-mkldnn", command)
        self.assertIn("--skip-warmup", command)

    def test_render_wrapping_preserves_every_reference_word(self):
        image = Image.new("RGB", (900, 1100), "white")
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)
        draw = ImageDraw.Draw(image)
        lines = wrap(draw, TEXT, font, 800)
        self.assertEqual(" ".join(lines).split(), TEXT.split())
        self.assertTrue(all(draw.textlength(line, font=font) <= 800 for line in lines))

    def test_public_selection_is_reproducible_and_excludes_ignored_regions(self):
        records = []
        for i, category in enumerate(("book", "newspaper", "research_report")):
            records.append({"page_info": {"image_path": f"{i}.jpg", "page_attribute": {"language": "english", "data_source": category}},
                "layout_dets": [{"anno_id": 1, "category_type": "text_block", "text": "word " * 30,
                                 "poly": [0, 0, 10, 0, 10, 10, 0, 10], "ignore": False},
                                {"anno_id": 2, "category_type": "text_block", "text": "ignored " * 30,
                                 "poly": [0, 0, 10, 0, 10, 10, 0, 10], "ignore": True}]})
        first = select_regions(records, 3)
        self.assertEqual(first, select_regions(records, 3))
        self.assertEqual(len(first), 3)
        self.assertTrue(all(block["anno_id"] == 1 for _, block in first))
        self.assertEqual(len({r["page_info"]["image_path"] for r, _ in first}), 3)

    def test_dataset_rejects_paths_outside_its_directory(self):
        folder = Path("eval/test-runs") / uuid4().hex
        folder.mkdir(parents=True)
        try:
            (folder / "dataset.json").write_text(json.dumps({"pages": [{"page": 1, "image": "../outside.png"}]}))
            with self.assertRaisesRegex(ValueError, "inside"):
                load_dataset(folder)
        finally:
            assert folder.resolve().parent == Path("eval/test-runs").resolve()
            shutil.rmtree(folder)

    def test_control_validation_does_not_invent_a_missing_prediction(self):
        folder = Path("eval/test-runs") / uuid4().hex
        folder.mkdir(parents=True)
        try:
            image = folder / "page_001.png"
            Image.new("RGB", (5, 5), "white").save(image)
            reference = folder / "page_001.txt"
            reference.write_text("known text")
            (folder / "dataset.json").write_text(json.dumps({"pages": [{"page": 1,
                "image": image.name, "reference": reference.name, "variant": "test",
                "image_sha256": sha256(image), "reference_sha256": sha256(reference)}]}))
            for engine in ("indra", "tesseract", "paddleocr"):
                (folder / engine / "run").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "Missing indra prediction"):
                validate(folder)
        finally:
            assert folder.resolve().parent == Path("eval/test-runs").resolve()
            shutil.rmtree(folder)


if __name__ == "__main__":
    unittest.main()
