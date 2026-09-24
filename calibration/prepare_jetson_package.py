"""Create a self-contained Jetson deployment package for the split ONNX model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf

IMAGE_SIZE = 256


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--onnx-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-input-dir", type=Path, default=root / "Experiment_test" / "test_2012AVFSI" / "V-FSI")
    parser.add_argument("--test-reference-dir", type=Path, default=root / "Experiment_test" / "test_2012AVFSI" / "original")
    parser.add_argument("--test-metrics", type=Path, default=root / "analysis_results" / "lighta_dual_highfreqlogmagphase40_mse_ssim10_w075_l2_rcca2_e50_foreground_showcase1176" / "per_image.csv")
    parser.add_argument("--calibration-count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=905)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def sample_id(path: Path) -> int:
    import re

    match = re.search(r"img_(\d+)", path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot extract img_<number> ID: {path}")
    return int(match.group(1))


def indexed_images(directory: Path) -> dict[int, Path]:
    images: dict[int, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            images[sample_id(path)] = path
    if not images:
        raise ValueError(f"No images in {directory}")
    return images


def paired(reference_dir: Path, input_dir: Path, limit: int = 0) -> list[tuple[int, Path, Path]]:
    refs, inputs = indexed_images(reference_dir), indexed_images(input_dir)
    if refs.keys() != inputs.keys():
        raise ValueError("Unmatched input/reference IDs")
    result = [(key, refs[key], inputs[key]) for key in sorted(refs)]
    return result[:limit] if limit else result


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"Expected 256x256: {path}")
        return np.asarray(image, dtype=np.float32) / 255.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(root: Path, run_dir: Path, config: dict, stats: np.ndarray) -> tf.keras.Model:
    sys.path.insert(0, str(root / "Modules"))
    network = importlib.import_module(f"frequency_spatial_network_{config['model_suffix'].lstrip('_')}")
    model = network.wnet(*(float(value) for value in stats), kshape=(5, 5), kshape2=(3, 3), **config["model_kwargs"])
    model.load_weights(str(run_dir / "best.hdf5"))
    return model


def main() -> int:
    args = parse_args()
    root, run_dir, onnx_dir, output_dir = Path(__file__).resolve().parent, args.run_dir.resolve(), args.onnx_dir.resolve(), args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite deployment package: {output_dir}")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((onnx_dir / "manifest.json").read_text(encoding="utf-8"))
    stats = np.asarray(config["normalization_stats"], dtype=np.float32)
    train_pairs = paired(Path(config["train_reference_dir"]), Path(config["train_input_dir"]), int(config["train_limit"]))
    if args.calibration_count > len(train_pairs):
        raise ValueError("Calibration count exceeds actual training samples")
    selected = [train_pairs[index] for index in sorted(np.random.default_rng(args.seed).choice(len(train_pairs), args.calibration_count, replace=False))]

    models_dir, calibration_dir, test_dir, scripts_dir = (output_dir / name for name in ("models", "calibration", "test_data", "scripts"))
    for directory in (models_dir, calibration_dir, test_dir, scripts_dir, calibration_dir / "images", test_dir / "input", test_dir / "reference"):
        directory.mkdir(parents=True, exist_ok=True)
    for name in ("frequency.onnx", "spatial.onnx", "manifest.json"):
        shutil.copy2(onnx_dir / name, models_dir / name)
    for name in ("jetson_build_int8.py", "jetson_infer_trt.py", "README_JETSON.md"):
        shutil.copy2(root / name, scripts_dir / name)

    frequency_path = calibration_dir / "frequency_input_float32.npy"
    spatial_path = calibration_dir / "spatial_input_float32.npy"
    frequency_data = np.lib.format.open_memmap(frequency_path, mode="w+", dtype=np.float32, shape=(len(selected), 256, 256, 2))
    spatial_data = np.lib.format.open_memmap(spatial_path, mode="w+", dtype=np.float32, shape=(len(selected), 256, 256, 1))
    model = load_model(root, run_dir, config, stats)
    mean, std, fft_scale = (float(manifest["normalization"][name]) for name in ("mean", "std", "fft_scale"))
    calibration_rows: list[dict[str, object]] = []
    for start in range(0, len(selected), args.batch_size):
        chunk = selected[start : start + args.batch_size]
        inputs = np.stack([load_gray(item[2]) for item in chunk])
        spectrum = np.fft.fft2(inputs) / fft_scale
        normalized = (np.stack((spectrum.real, spectrum.imag), axis=-1).astype(np.float32) - mean) / std
        frequency_data[start : start + len(chunk)] = normalized
        pred_kspace = model(normalized, training=False)[0].numpy()
        physical = pred_kspace[..., 0] * std + mean + 1j * (pred_kspace[..., 1] * std + mean)
        spatial_data[start : start + len(chunk), ..., 0] = (np.fft.ifft2(physical).real * IMAGE_SIZE).astype(np.float32)
        for offset, (key, reference, source) in enumerate(chunk):
            destination = calibration_dir / "images" / source.name
            shutil.copy2(source, destination)
            calibration_rows.append({"calibration_index": start + offset, "sample_id": key, "input_file": f"images/{source.name}", "reference_file": reference.name, "selection": f"deterministic_random_seed_{args.seed}_from_first_{len(train_pairs)}_training_pairs"})
    del frequency_data, spatial_data

    with (calibration_dir / "calibration_index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(calibration_rows[0]))
        writer.writeheader(); writer.writerows(calibration_rows)

    test_pairs = paired(args.test_reference_dir.resolve(), args.test_input_dir.resolve())
    saved_metrics = {int(row["sample_id"]): row for row in csv.DictReader(args.test_metrics.resolve().open(encoding="utf-8"))}
    test_rows: list[dict[str, object]] = []
    for key, reference, source in test_pairs:
        shutil.copy2(source, test_dir / "input" / source.name)
        shutil.copy2(reference, test_dir / "reference" / reference.name)
        row: dict[str, object] = {"sample_id": key, "input_file": f"input/{source.name}", "reference_file": f"reference/{reference.name}"}
        row.update({f"highfreq_{name}": value for name, value in saved_metrics.get(key, {}).items() if name not in {"sample_id", "input", "reference"}})
        test_rows.append(row)
    fields = sorted({field for row in test_rows for field in row})
    with (test_dir / "test_data_index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(test_rows)
    shutil.copy2(args.test_metrics.resolve(), test_dir / "highfreq_showcase_per_image.csv")

    package_manifest = {
        "package": "lighta_dual_rcca2_e50_jetson_int8",
        "source_run": str(run_dir),
        "calibration": {"sample_count": len(selected), "seed": args.seed, "source": "random subset of the 6,400 training pairs used by this run", "frequency_tensor": frequency_path.name, "spatial_tensor": spatial_path.name},
        "test_data": {"sample_count": len(test_pairs), "source": "latest selected high-quality showcase", "index": "test_data/test_data_index.csv"},
        "model_sha256": {name: sha256(models_dir / name) for name in ("frequency.onnx", "spatial.onnx", "manifest.json")},
    }
    (output_dir / "package_manifest.json").write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(package_manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
