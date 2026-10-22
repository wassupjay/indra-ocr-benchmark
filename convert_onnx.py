"""Export the cached Paddle weights using the matching isolated native converter."""

import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

from evaluate import sha256
from benchmark import save_json


def unpack_wheel(wheel, target, expected_hash):
    if sha256(wheel) != expected_hash:
        raise ValueError(f"Wheel checksum mismatch: {wheel.name}")
    target = target.resolve()
    with zipfile.ZipFile(wheel) as package:
        if not all((target / name).resolve().is_relative_to(target) for name in package.namelist()):
            raise ValueError("Wheel contains a path outside its target")
        target.mkdir(parents=True, exist_ok=True)
        package.extractall(target)


def main():
    root = Path(__file__).resolve().parent
    for filename, directory, digest in (
        ("paddle2onnx-2.1.0-cp311-cp311-win_amd64.whl", ".onnx-tools", "478993e17ed0212b79a4d6e2d8d0582ebb19c7230b7f365d51222833e98581b3"),
        ("onnx-1.17.0-cp311-cp311-win_amd64.whl", ".onnx-tools", "95c03e38671785036bb704c30cd2e150825f6ab4763df3a4f1d249da48525957"),
        ("paddlepaddle-3.0.0.dev20250426-cp311-cp311-win_amd64.whl", ".onnx-paddle-3.0", "f62aaab2bd8d3ad4f4f7781bdeed43403546057b7afcdc10a4b33847b2617f1f"),
    ):
        unpack_wheel(root / ".package-cache" / filename, root / directory, digest)
    sys.path.insert(0, str(root / ".onnx-tools"))
    paddle = root / ".onnx-paddle-3.0/paddle"
    # The native export API is the same API called by paddle2onnx.convert.export.
    # Loading it directly avoids importing the old Paddle Python wrapper.
    dll_handles = [os.add_dll_directory(str(paddle / name)) for name in ("libs", "base")]
    library = ctypes.WinDLL(str(paddle / "base/libpaddle.pyd"))
    extension = next((root / ".onnx-tools/paddle2onnx").glob("*.pyd"))
    spec = importlib.util.spec_from_file_location("paddle2onnx_cpp2py_export", extension)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    import onnx

    baseline = json.loads((root / "publication/paddle-verification.json").read_text(encoding="utf-8"))
    output = root / ".onnx-models/PP-OCRv6_medium"
    provenance = {"converter": "Paddle2ONNX 2.1.0 native export API", "converter_paddle": "3.0.0.dev20250426",
        "opset_requested": 19, "auto_upgrade_opset": True, "onnx_checker": True,
        "fp16": False, "quantized": False, "python_post_optimizer": None,
        "baseline_results_sha256": baseline["results_sha256"],
        "wheel_sha256": {p.name: sha256(p) for pattern in ("paddle2onnx-2.1.0-*.whl", "paddlepaddle-3.0.0.dev20250426-*.whl", "onnx-1.17.0-*.whl") for p in (root / ".package-cache").glob(pattern)},
        "models": {}}
    for component, suffix in (("detector", "det"), ("recognizer", "rec")):
        source = Path.home() / f".paddlex/official_models/PP-OCRv6_medium_{suffix}"
        hashes = {p.name: sha256(p) for p in source.glob("inference.*") if p.is_file()}
        if hashes != baseline["metadata"]["model_hashes"][component]:
            raise ValueError(f"{component} weights/config changed since the measured baseline")
        destination = output / component
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Converting {component}", flush=True)
        data = converter.export(str(source / "inference.json"), str(source / "inference.pdiparams"),
            19, True, False, True, True, True, {}, "onnxruntime", "", "", False)
        if not data:
            raise RuntimeError(f"{component} converter returned an empty model")
        model_path = destination / "inference.onnx"
        model_path.write_bytes(data)
        onnx.checker.check_model(str(model_path), full_check=True)
        shutil.copyfile(source / "inference.yml", destination / "inference.yml")
        if hashes != {p.name: sha256(p) for p in source.glob("inference.*") if p.is_file()}:
            raise ValueError("Source model files changed during conversion")
        model = onnx.load(str(model_path), load_external_data=False)
        provenance["models"][component] = {"source_sha256": hashes, "onnx_sha256": sha256(model_path),
            "opsets": [{"domain": x.domain, "version": x.version} for x in model.opset_import],
            "onnx_full_check_passed": True}
        save_json(output / "conversion.json", provenance)
        print(f"Validated {model_path} ({len(data)} bytes)", flush=True)
    print("Both converted models passed full ONNX validation.", flush=True)


if __name__ == "__main__":
    main()
