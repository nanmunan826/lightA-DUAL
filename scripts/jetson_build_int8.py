"""Build one static INT8 TensorRT engine from ONNX and a float32 .npy calibration tensor.

Run this script on the target Jetson, not on Windows.  It requires the
JetPack TensorRT Python package and ``pycuda``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401 - creates the CUDA context used by TensorRT
import pycuda.driver as cuda
import tensorrt as trt


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, samples: Path, cache: Path):
        super().__init__()
        self.samples = np.load(samples, mmap_mode="r")
        if self.samples.dtype != np.float32 or self.samples.ndim != 4:
            raise ValueError("Calibration tensor must be N,H,W,C float32")
        self.cache, self.index = cache, 0
        self.device = cuda.mem_alloc(self.samples[0:1].nbytes)

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names):
        if self.index >= len(self.samples):
            return None
        cuda.memcpy_htod(self.device, np.ascontiguousarray(self.samples[self.index : self.index + 1]))
        self.index += 1
        return [int(self.device)]

    def read_calibration_cache(self):
        return self.cache.read_bytes() if self.cache.is_file() else None

    def write_calibration_cache(self, cache):
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache.write_bytes(cache)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workspace-mib", type=int, default=1024)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024 * 1024)
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if not builder.platform_has_fast_int8:
        raise RuntimeError("This Jetson TensorRT installation does not expose fast INT8")
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = EntropyCalibrator(args.calibration, args.cache)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the INT8 engine")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(serialized)
    print(f"Wrote {args.engine} ({args.engine.stat().st_size / 1024 / 1024:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
