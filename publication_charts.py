"""Render three source-backed editorial bar charts using the existing Pillow stack."""

import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "publication"
ORDER = ["indra", "tesseract", "paddleocr_tiny", "easyocr"]
LABELS = ["Indra", "Tesseract", "PaddleOCR Tiny", "EasyOCR"]
COLORS = ["#21618a", "#b67c28", "#287f76", "#8f6595"]
BG, INK, MUTED = "#fbfaf6", "#152d36", "#53646c"
FONT = "C:/Windows/Fonts/arial.ttf"
BOLD = "C:/Windows/Fonts/arialbd.ttf"
W, H, LEFT, RIGHT = 1800, 1100, 370, 1600


def text(draw, xy, value, size=30, fill=INK, bold=False):
    font = ImageFont.truetype(BOLD if bold else FONT, size)
    bounds = draw.textbbox(xy, value, font=font)
    assert bounds[0] >= 0 and bounds[2] <= W and bounds[3] <= H, value
    draw.text(xy, value, font=font, fill=fill)


def base(title, subtitle, maximum, ticks, unit):
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (70, 45), "CPU OCR BENCHMARK / 114 PAGES", 25, MUTED, True)
    text(draw, (70, 100), title, 54, bold=True)
    text(draw, (70, 180), subtitle, 29, MUTED)
    for tick in ticks:
        x = LEFT + (RIGHT - LEFT) * tick / maximum
        draw.line((x, 285, x, 850), fill="#dde2e0", width=2)
        text(draw, (x - 13, 867), f"{tick:g}", 25, MUTED)
    text(draw, (LEFT, 915), unit + " / linear scale starting at zero", 26, MUTED)
    return image, draw


def single(filename, title, subtitle, values, maximum, ticks, unit, display, notes):
    assert len(values) == 4 and all(math.isfinite(v) and 0 <= v <= maximum for v in values)
    image, draw = base(title, subtitle, maximum, ticks, unit)
    for i, (label, color, value) in enumerate(zip(LABELS, COLORS, values)):
        y = 320 + i * 137
        text(draw, (70, y + 13), label, 35, bold=True)
        end = LEFT + (RIGHT - LEFT) * value / maximum
        draw.rectangle((LEFT, y, end, y + 68), fill=color)
        text(draw, (end + 17, y + 14), display(value), 30, bold=True)
    for i, note in enumerate(notes):
        text(draw, (70, 982 + i * 40), note, 25, MUTED)
    image.save(OUT / filename)


def main():
    source_files = [OUT / "cpu-four-comparison.json", OUT / "transcription-comparison.json"]
    current, candidate = [json.loads(p.read_text(encoding="utf-8")) for p in source_files]
    assert current["completed"]
    models = current["statistics"]["models"]
    assert set(models) == set(ORDER)
    assert candidate["full_book_provisional"]["pages"] == 114
    speed = [models[e]["pages_per_minute"] for e in ORDER]
    memory = [current["resources"][e]["sampled_peak_tree_rss_mib"] / 1024 for e in ORDER]
    pdf_wer = [models[e]["wer"] * 100 for e in ORDER]
    candidate_wer = [candidate["full_book_provisional"]["engines"][e]["wer"] * 100 for e in ORDER]
    single("chart-speed.png", "How fast did each engine read the book?", "Pages per minute / higher is faster / median of three complete passes",
        speed, 240, [0, 50, 100, 150, 200], "Pages per minute", lambda v: f"{v:.2f}",
        ["Same saved images; CPU only. Rendering, initial loading and warmup excluded.",
         "Tesseract CLI startup included. Earlier Indra timing drift remains unexplained."])
    single("chart-memory.png", "How much RAM did each worker use?", "Peak sampled process-tree memory / lower uses less RAM",
        memory, 11, [0, 2, 4, 6, 8, 10], "GiB (1024 MiB)", lambda v: f"{v:.2f} GiB",
        ["Whole worker, including loading and warmup; sampled every 100 ms.",
         "Summed RSS can double-count shared memory; brief peaks may be missed."])
    image, draw = base("How closely did the recognized words match?", "Word error rate (%) / lower is closer agreement / both references are unverified", 20, [0, 5, 10, 15, 20], "Word error rate (%)")
    draw.rectangle((LEFT, 235, LEFT + 30, 261), fill=INK)
    text(draw, (LEFT + 43, 230), "PDF text layer", 25)
    draw.rectangle((760, 235, 790, 261), fill=BG, outline=INK, width=3)
    text(draw, (803, 230), "Supplied AI-assisted transcription", 25)
    for i, (label, color, a, b) in enumerate(zip(LABELS, COLORS, pdf_wer, candidate_wer)):
        y = 317 + i * 137
        text(draw, (70, y + 21), label, 35, bold=True)
        for offset, value, outlined in [(0, a, False), (49, b, True)]:
            assert math.isfinite(value) and 0 <= value <= 20
            end = LEFT + (RIGHT - LEFT) * value / 20
            draw.rectangle((LEFT, y + offset, end, y + offset + 35), fill=BG if outlined else color, outline=color, width=3)
            text(draw, (end + 15, y + offset + 1), f"{value:.3f}%", 26, bold=True)
    text(draw, (70, 982), "All 114 pages. Case and punctuation count; these are not verified accuracy scores.", 25, MUTED)
    text(draw, (70, 1022), "Transcription uncertainty markers use supplied best readings. No OCR rerun.", 25, MUTED)
    image.save(OUT / "chart-word-errors.png")
    evidence = {"source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "engine_order": ORDER, "speed_pages_per_minute": speed, "peak_memory_gib": memory,
        "pdf_reference_wer_percent": pdf_wer, "candidate_reference_wer_percent": candidate_wer,
        "chart_axes": "Linear, zero baseline; no truncated axes. Memory converted from MiB to GiB by dividing by 1024.",
        "reference_status": "Both candidate references unverified; no new inference measurements."}
    (OUT / "chart-data.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print("Rendered and bounds-checked three charts; numeric sources recorded in chart-data.json.")


if __name__ == "__main__":
    main()
