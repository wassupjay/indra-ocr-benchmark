"""Prepare a reproducible English text-region subset of OmniDocBench.

Uses annotated crops to avoid inventing a full-page transcript or reading order.
These are custom region CER/WER scores, not official OmniDocBench scores.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import re
import urllib.parse
import urllib.request

from PIL import Image

from benchmark import save_json
from evaluate import sha256


REPO = "opendatalab/OmniDocBench"


def select_regions(records, count):
    rng = random.Random(20261018)
    groups = defaultdict(list)
    seen = set()
    for record in sorted(records, key=lambda r: r["page_info"]["image_path"]):
        info = record["page_info"]
        attrs = info.get("page_attribute", {})
        if attrs.get("language") != "english" or attrs.get("data_source") == "note":
            continue
        if info["image_path"] in seen:
            continue
        eligible = []
        for block in record["layout_dets"]:
            attributes = block.get("attribute", {})
            text = block.get("text", "")
            spans = block.get("line_with_spans", [])
            if (block.get("category_type") == "text_block" and not block.get("ignore", False)
                and not block.get("merge_list") and 20 <= len(text.split()) <= 100
                and attributes.get("text_rotate", "normal") == "normal"
                and attributes.get("text_language", "text_en") == "text_en"
                and not any(span.get("category_type") in ("equation_inline", "equation_ignore") for span in spans)
                and len(block.get("poly", [])) == 8):
                eligible.append(block)
        if eligible:
            eligible.sort(key=lambda b: b.get("anno_id", 0))
            groups[attrs.get("data_source", "unknown")].append((record, rng.choice(eligible)))
            seen.add(info["image_path"])
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < count and any(groups.values()):
        for name in sorted(groups):
            if groups[name] and len(selected) < count:
                selected.append(groups[name].pop())
    if not selected:
        raise ValueError("No eligible English text blocks. Check the downloaded annotation schema.")
    return selected


def download(url, destination):
    if destination.exists():
        return
    request = urllib.request.Request(url, headers={"User-Agent": "indraocr-benchmark/1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def prepare(source, count, revision, fetch):
    annotation = source / "OmniDocBench.json"
    records = json.loads(annotation.read_text(encoding="utf-8"))
    selected = select_regions(records, count)
    revision_tag = revision[:12] if re.fullmatch(r"[0-9a-f]{40,64}", revision) else "local"
    output = Path("eval/benchmarks") / f"omnidocbench-text-regions-{len(selected)}-{sha256(annotation)[:12]}-{revision_tag}"
    if (output / "dataset.json").exists():
        existing = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
        for page in existing["pages"]:
            for key in ("image", "reference"):
                if sha256(output / page[key]) != page[f"{key}_sha256"]:
                    raise ValueError("Previously prepared public subset changed.")
        return output
    output.mkdir(parents=True, exist_ok=True)
    pages = []
    for number, (record, block) in enumerate(selected, 1):
        info = record["page_info"]
        name = Path(info["image_path"]).name
        if name in ("", ".", "..") or Path(name).suffix.lower() not in (".png", ".jpg", ".jpeg"):
            raise ValueError("Invalid source image filename.")
        image_source = source / "images" / name
        if fetch:
            download(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/images/{urllib.parse.quote(name)}", image_source)
        with Image.open(image_source) as image:
            if image.size != (info["width"], info["height"]):
                raise ValueError(f"Annotation/image dimensions differ for {name}; do not silently rescale annotations.")
            coordinates = block["poly"]
            box = (math.floor(min(coordinates[::2])), math.floor(min(coordinates[1::2])),
                   math.ceil(max(coordinates[::2])), math.ceil(max(coordinates[1::2])))
            if not (0 <= box[0] < box[2] <= image.width and 0 <= box[1] < box[3] <= image.height):
                raise ValueError(f"Invalid annotated crop in {name}")
            image_path = output / f"page_{number:03}.png"
            image.crop(box).convert("RGB").save(image_path)
        reference = output / f"page_{number:03}.txt"
        reference.write_text(block["text"], encoding="utf-8")
        pages.append({"page": number, "image": image_path.name, "reference": reference.name,
            "image_sha256": sha256(image_path), "reference_sha256": sha256(reference),
            "source_image": name, "source_image_sha256": sha256(image_source),
            "source_anno_id": block["anno_id"], "crop": box,
            "data_source": info["page_attribute"].get("data_source")})
    save_json(output / "dataset.json", {"kind": "public_annotated_text_regions", "unit": "text regions",
        "source": f"https://huggingface.co/datasets/{REPO}", "source_revision": revision,
        "annotation_sha256": sha256(annotation), "seed": 20261018,
        "reference_status": "dataset-provided text annotations; not a PDF text layer",
        "scope": "one nonignored English text_block of 20–100 words per selected page, no merged/inline-equation blocks, round-robin document categories; gold crop boundaries supplied to all models",
        "limitation": "custom crop recognition evaluation; excludes page detection/reading order; not the official OmniDocBench score; annotation errors and training-set overlap remain possible",
        "pages": pages})
    return output


def fetch_source(source):
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}", timeout=15) as response:
        revision = json.load(response)["sha"]
    if not re.fullmatch(r"[0-9a-f]{40,64}", revision):
        raise ValueError("Dataset API returned an invalid source revision.")
    source = source / revision
    download(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/OmniDocBench.json", source / "OmniDocBench.json")
    return source, revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("eval/public-source/OmniDocBench"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")
    revision = "local snapshot; see annotation hash"
    try:
        if args.download:
            args.source, revision = fetch_source(args.source)
        output = prepare(args.source, args.count, revision, args.download)
        print(f"Prepared {output}")
        print(f'Run: python benchmark.py --dataset "{output}"')
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Public dataset preparation failed: {error}\nNo public-dataset scores have been produced.\n")


if __name__ == "__main__":
    main()
