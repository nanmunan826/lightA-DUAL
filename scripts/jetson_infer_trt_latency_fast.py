"""Run the two-engine TensorRT dual-domain pipeline on Jetson.

Requires the JetPack TensorRT Python API and pycuda.  IFFT is intentionally
kept in FP32 outside TensorRT to preserve the model's training contract.

Extended with:
  - End-to-end timing with / without I/O
  - Optional FP32 TensorRT baseline accuracy comparison
  - Warmup, image preloading, async PNG save (compress_level=1)
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
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
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="int8")
    parser.add_argument("--baseline-fp32", action="store_true",
                        help="Compare against FP32 TensorRT engines")
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


def compute_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    pred = pred.astype(np.float64).ravel()
    ref = ref.astype(np.float64).ravel()
    diff = pred - ref
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    max_abs = np.max(np.abs(diff))
    dot = np.dot(pred, ref)
    norm_pred = np.linalg.norm(pred)
    norm_ref = np.linalg.norm(ref)
    cos_sim = dot / (norm_pred * norm_ref + 1e-12)
    return {"max_abs": float(max_abs), "rmse": float(rmse), "cos_sim": float(cos_sim)}


def compute_psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    mse = np.mean((pred - ref) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def percentile_str(arr_ms: np.ndarray) -> str:
    return (f"mean={arr_ms.mean():7.2f} ms | "
            f"min={arr_ms.min():7.2f} | max={arr_ms.max():7.2f} | "
            f"P50={np.percentile(arr_ms, 50):7.2f} | P99={np.percentile(arr_ms, 99):7.2f}")


def main() -> int:
    args = parse_args()
    package = args.package_dir.resolve()
    manifest = json.loads((package / "models" / "manifest.json").read_text(encoding="utf-8"))
    mean, std, fft_scale = (float(manifest["normalization"][name]) for name in ("mean", "std", "fft_scale"))
    engines = package / "engines"
    frequency = EngineRunner(engines / f"frequency_{args.precision}.engine")
    spatial = EngineRunner(engines / f"spatial_{args.precision}.engine")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ========== FP32 基准引擎初始化（可选） ==========
    baseline_freq = None
    baseline_spatial = None
    if args.baseline_fp32 and args.precision != "fp32":
        baseline_freq = EngineRunner(engines / "frequency_fp32.engine")
        baseline_spatial = EngineRunner(engines / "spatial_fp32.engine")
        print("[INFO] FP32 baseline engines loaded.")
    elif args.baseline_fp32 and args.precision == "fp32":
        print("[WARN] precision=fp32, baseline comparison skipped (same as target).")

    # ========== 优化 1：图片预加载到内存 ==========
    image_paths = inputs(args.input.resolve())
    print(f"[INFO] Preloading {len(image_paths)} images into memory...")
    image_cache = []
    read_times = []
    for p in image_paths:
        t0 = time.perf_counter()
        img = load_image(p)
        read_times.append(time.perf_counter() - t0)
        image_cache.append((p, img))
    print(f"[INFO] Preload done. Avg read: {np.mean(read_times)*1000:.2f} ms/image, "
          f"total: {sum(read_times):.2f} s")

    # ========== 优化 2：预热 3 张图 ==========
    print("[INFO] Warming up with 3 images...")
    for idx in range(min(3, len(image_cache))):
        _, source = image_cache[idx]
        spectrum = np.fft.fft2(source) / fft_scale
        normalized = (np.stack((spectrum.real, spectrum.imag), axis=-1)[None].astype(np.float32) - mean) / std
        pk = frequency(normalized)
        phy = pk[0, ..., 0] * std + mean + 1j * (pk[0, ..., 1] * std + mean)
        fi = (np.fft.ifft2(phy).real * 256)[None, ..., None].astype(np.float32)
        _ = spatial(fi)
    print("[INFO] Warmup done.")

    # ========== 计时容器 ==========
    times = {
        "read_image": read_times,     # 来自预加载阶段
        "preprocess": [],
        "frequency": [],
        "ifft": [],
        "spatial": [],
        "save_submit": [],
        "save_actual": [],
        "base_frequency": [],
        "base_ifft": [],
        "base_spatial": [],
    }
    acc_metrics = {
        "freq_max_abs": [], "freq_rmse": [], "freq_cos_sim": [],
        "img_psnr": [], "img_mse": [],
    }
    save_actual_by_index = [None] * len(image_cache)
    save_lock = threading.Lock()

    # ========== 优化 3 + 4：异步 PNG 保存 + compress_level=1 ==========
    def save_png_async(prediction: np.ndarray, path: Path, idx: int) -> None:
        t0 = time.perf_counter()
        Image.fromarray(np.round(np.clip(prediction, 0, 1) * 255).astype(np.uint8)).save(
            path, format="PNG", compress_level=1)
        elapsed = time.perf_counter() - t0
        with save_lock:
            save_actual_by_index[idx] = elapsed

    executor = ThreadPoolExecutor(max_workers=4)

    # ========== 主循环 ==========
    for idx, (source_path, source) in enumerate(image_cache):
        # 1. 预处理（不含读图）
        t0 = time.perf_counter()
        spectrum = np.fft.fft2(source) / fft_scale
        normalized = (np.stack((spectrum.real, spectrum.imag), axis=-1)[None].astype(np.float32) - mean) / std
        times["preprocess"].append(time.perf_counter() - t0)

        # 2. frequency 推理
        t0 = time.perf_counter()
        predicted_kspace = frequency(normalized)
        times["frequency"].append(time.perf_counter() - t0)

        # 3. 反归一化 + IFFT
        t0 = time.perf_counter()
        physical = predicted_kspace[0, ..., 0] * std + mean + 1j * (predicted_kspace[0, ..., 1] * std + mean)
        frequency_image = (np.fft.ifft2(physical).real * 256)[None, ..., None].astype(np.float32)
        times["ifft"].append(time.perf_counter() - t0)

        # 4. spatial 推理
        t0 = time.perf_counter()
        prediction = spatial(frequency_image)[0, ..., 0]
        times["spatial"].append(time.perf_counter() - t0)

        # 5. 提交异步保存（仅记录提交时间）
        t0 = time.perf_counter()
        executor.submit(
            save_png_async,
            prediction,
            args.output_dir / f"{source_path.stem}_reconstruction.png",
            idx,
        )
        times["save_submit"].append(time.perf_counter() - t0)

        # 6. FP32 基准对比（可选）
        if baseline_freq is not None and baseline_spatial is not None:
            t0 = time.perf_counter()
            base_freq_out = baseline_freq(normalized)
            times["base_frequency"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            base_physical = base_freq_out[0, ..., 0] * std + mean + 1j * (base_freq_out[0, ..., 1] * std + mean)
            base_freq_image = (np.fft.ifft2(base_physical).real * 256)[None, ..., None].astype(np.float32)
            times["base_ifft"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            base_pred = baseline_spatial(base_freq_image)[0, ..., 0]
            times["base_spatial"].append(time.perf_counter() - t0)

            fm = compute_metrics(predicted_kspace, base_freq_out)
            acc_metrics["freq_max_abs"].append(fm["max_abs"])
            acc_metrics["freq_rmse"].append(fm["rmse"])
            acc_metrics["freq_cos_sim"].append(fm["cos_sim"])

            psnr = compute_psnr(prediction, base_pred)
            mse = float(np.mean((prediction - base_pred) ** 2))
            acc_metrics["img_psnr"].append(psnr)
            acc_metrics["img_mse"].append(mse)

    # 等待所有异步保存完成
    executor.shutdown(wait=True)

    # 收集真实保存耗时
    times["save_actual"] = [t for t in save_actual_by_index if t is not None]

    # ========== 优化 5：计算三组总时间 ==========
    n = len(times["preprocess"])
    total_with_io, total_no_io, total_async = [], [], []

    for i in range(n):
        pre = times["preprocess"][i]
        freq = times["frequency"][i]
        ift = times["ifft"][i]
        spa = times["spatial"][i]
        submit = times["save_submit"][i]
        read = times["read_image"][i] if i < len(times["read_image"]) else 0.0
        save_act = save_actual_by_index[i] if save_actual_by_index[i] is not None else 0.0

        total_no_io.append(pre + freq + ift + spa)
        total_async.append(read + pre + freq + ift + spa + submit)
        total_with_io.append(read + pre + freq + ift + spa + save_act)

    times["total_with_io"] = total_with_io
    times["total_no_io"] = total_no_io
    times["total_async"] = total_async

    # ========== 输出计时报告 ==========
    print("\n" + "=" * 60)
    print(f"Pipeline Benchmark Report (precision={args.precision})")
    print("=" * 60)

    print("[单阶段耗时]")
    for key in ["read_image", "preprocess", "frequency", "ifft", "spatial", "save_submit", "save_actual"]:
        arr = np.array(times[key]) * 1000
        if len(arr) == 0:
            continue
        print(f"{key:14s}: {percentile_str(arr)}")

    if times["base_frequency"]:
        print("-" * 60)
        print("[FP32 基准单阶段耗时]")
        for key in ["base_frequency", "base_ifft", "base_spatial"]:
            arr = np.array(times[key]) * 1000
            print(f"{key:14s}: {percentile_str(arr)}")

    print("-" * 60)
    print("[总延迟对比]")
    for key, label in [("total_with_io", "with_io  (含读图+实际保存)"),
                        ("total_no_io",   "no_io    (纯计算)"),
                        ("total_async",   "async    (主线程视角)")]:
        arr = np.array(times[key]) * 1000
        print(f"{label:28s}: {percentile_str(arr)}")

    # ========== 吞吐量 ==========
    print("=" * 60)
    print(f"Throughput (with_io): {1 / np.mean(times['total_with_io']):.2f} FPS")
    print(f"Throughput (no_io)  : {1 / np.mean(times['total_no_io']):.2f} FPS")
    print(f"Throughput (async)  : {1 / np.mean(times['total_async']):.2f} FPS")
    print(f"Samples: {n}")
    print("=" * 60)

    # ========== FP32 加速比（基于 no_io） ==========
    if times["base_frequency"]:
        base_total = (np.array(times["base_frequency"]) +
                      np.array(times["base_ifft"]) +
                      np.array(times["base_spatial"]))
        speedup = np.mean(base_total) / np.mean(times["total_no_io"])
        print(f"Speedup (TRT_fp32 vs TRT_{args.precision}, no_io): {speedup:.2f}x")
        print("=" * 60)

    # ========== 精度报告 ==========
    if acc_metrics["freq_max_abs"]:
        print("\n" + "=" * 60)
        print(f"Accuracy Report vs FP32 baseline (precision={args.precision})")
        print("=" * 60)
        print(f"Frequency output:")
        print(f"  Max Abs Error : {np.mean(acc_metrics['freq_max_abs']):.6f}")
        print(f"  RMSE          : {np.mean(acc_metrics['freq_rmse']):.6f}")
        print(f"  Cosine Sim    : {np.mean(acc_metrics['freq_cos_sim']):.6f}")
        print(f"Final image:")
        print(f"  PSNR (dB)     : {np.mean(acc_metrics['img_psnr']):.4f}")
        print(f"  MSE           : {np.mean(acc_metrics['img_mse']):.6f}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
