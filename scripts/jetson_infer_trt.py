"""Run the two-engine TensorRT dual-domain pipeline on Jetson.

Requires the JetPack TensorRT Python API and pycuda.  IFFT is intentionally
kept in FP32 outside TensorRT to preserve the model's training contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt


class EngineRunner:
    def __init__(self, engine_path: Path):
        logger = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors) if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]
        self.outputs = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors) if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise ValueError("Expected exactly one input and one output")

    def __call__(self, array: np.ndarray) -> np.ndarray:
        value = np.ascontiguousarray(array.astype(np.float32, copy=False))
        self.context.set_input_shape(self.inputs[0], value.shape)
        output_shape = tuple(self.context.get_tensor_shape(self.outputs[0]))
        output = np.empty(output_shape, dtype=np.float32)
        device_input, device_output = cuda.mem_alloc(value.nbytes), cuda.mem_alloc(output.nbytes)
        cuda.memcpy_htod_async(device_input, value, self.stream)
        self.context.set_tensor_address(self.inputs[0], int(device_input))
        self.context.set_tensor_address(self.outputs[0], int(device_output))
        if not self.context.execute_async_v3(self.stream.handle):
            raise RuntimeError("TensorRT execution failed")
        cuda.memcpy_dtoh_async(output, device_output, self.stream)
        self.stream.synchronize()
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="One image or a directory of images")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp16", "fp32","int8"), default="int8")
    return parser.parse_args()


def inputs(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png"})


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (256, 256):
            raise ValueError(f"Expected 256x256 image: {path}")
        return np.asarray(image, dtype=np.float32) / 255.0


def main() -> int:
    args = parse_args()
    package = args.package_dir.resolve()
    manifest = json.loads((package / "models" / "manifest.json").read_text(encoding="utf-8"))
    mean, std, fft_scale = (float(manifest["normalization"][name]) for name in ("mean", "std", "fft_scale"))
    engines = package / "engines"
    frequency = EngineRunner(engines / f"frequency_{args.precision}.engine")
    spatial = EngineRunner(engines / f"spatial_{args.precision}.engine")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source_path in inputs(args.input.resolve()):
        source = load_image(source_path)
        spectrum = np.fft.fft2(source) / fft_scale
        normalized = (np.stack((spectrum.real, spectrum.imag), axis=-1)[None].astype(np.float32) - mean) / std
        predicted_kspace = frequency(normalized)
        physical = predicted_kspace[0, ..., 0] * std + mean + 1j * (predicted_kspace[0, ..., 1] * std + mean)
        frequency_image = (np.fft.ifft2(physical).real * 256)[None, ..., None].astype(np.float32)
        prediction = spatial(frequency_image)[0, ..., 0]
        Image.fromarray(np.round(np.clip(prediction, 0, 1) * 255).astype(np.uint8)).save(args.output_dir / f"{source_path.stem}_reconstruction.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
