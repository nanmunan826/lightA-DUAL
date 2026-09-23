"""Run the two-engine TensorRT dual-domain pipeline on Jetson.

Requires the JetPack TensorRT Python API and pycuda.  IFFT is intentionally
kept in FP32 outside TensorRT to preserve the model's training contract.

Extended with end-to-end timing and optional ONNX accuracy comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

# 可选：ONNX Runtime 用于精度基准对比
try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False


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
    parser.add_argument("--precision", choices=("fp16", "int8"), default="int8")
    # 可选 ONNX 基准路径，用于精度对比和加速比计算
    parser.add_argument("--onnx-frequency", type=Path, default=None,
                        help="Path to frequency.onnx for accuracy baseline")
    parser.add_argument("--onnx-spatial", type=Path, default=None,
                        help="Path to spatial.onnx for accuracy baseline")
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


# ================= ONNX 推理辅助函数 =================
def create_onnx_session(onnx_path: Path):
    """创建 ONNX Runtime 会话，优先使用 CUDA，否则用 CPU。"""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in ort.get_available_providers() else ['CPUExecutionProvider']
    return ort.InferenceSession(str(onnx_path), providers=providers)


def run_onnx(session, input_array: np.ndarray) -> np.ndarray:
    """运行 ONNX 模型，返回输出数组。"""
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_array})[0]


def compute_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    """计算预测与基准之间的误差指标。"""
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
    """计算 PSNR，输入范围假设为 [0,1]。"""
    mse = np.mean((pred - ref) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)
# ============================================================


def main() -> int:
    args = parse_args()
    package = args.package_dir.resolve()
    manifest = json.loads((package / "models" / "manifest.json").read_text(encoding="utf-8"))
    mean, std, fft_scale = (float(manifest["normalization"][name]) for name in ("mean", "std", "fft_scale"))
    engines = package / "engines"
    frequency = EngineRunner(engines / f"frequency_{args.precision}.engine")
    spatial = EngineRunner(engines / f"spatial_{args.precision}.engine")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 初始化 ONNX 基准（仅当提供了两个路径且 onnxruntime 可用） ==========
    onnx_freq_sess = None
    onnx_spatial_sess = None
    if args.onnx_frequency is not None and args.onnx_spatial is not None:
        if not _HAS_ORT:
            print("[WARN] onnxruntime not installed, skipping accuracy/benchmark comparison.")
        else:
            onnx_freq_sess = create_onnx_session(args.onnx_frequency)
            onnx_spatial_sess = create_onnx_session(args.onnx_spatial)
            print("[INFO] ONNX baseline sessions created.")
    # =============================================================================

    # ========== 计时与精度收集容器 ==========
    times = {
        "preprocess": [],
        "frequency": [],
        "ifft": [],
        "spatial": [],
        "postprocess": [],
        "total": [],
        # ONNX 计时（仅当启用 ONNX 基准时才有数据）
        "onnx_frequency": [],
        "onnx_ifft": [],
        "onnx_spatial": [],
        "onnx_total": [],
    }
    acc_metrics = {
        "freq_max_abs": [], "freq_rmse": [], "freq_cos_sim": [],
        "img_psnr": [], "img_mse": [],
    }
    # ==========================================

    for source_path in inputs(args.input.resolve()):
        t_total_start = time.perf_counter()

        # 1. 预处理
        t0 = time.perf_counter()
        source = load_image(source_path)
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

        # 5. 保存
        t0 = time.perf_counter()
        Image.fromarray(np.round(np.clip(prediction, 0, 1) * 255).astype(np.uint8)).save(
            args.output_dir / f"{source_path.stem}_reconstruction.png")
        times["postprocess"].append(time.perf_counter() - t0)

        times["total"].append(time.perf_counter() - t_total_start)

        # ========== ONNX 推理（计时 + 精度对比，仅当启用时执行） ==========
        if onnx_freq_sess is not None and onnx_spatial_sess is not None:
            t_onnx_total_start = time.perf_counter()

            # ONNX frequency 推理
            t0 = time.perf_counter()
            onnx_freq_out = run_onnx(onnx_freq_sess, normalized)
            times["onnx_frequency"].append(time.perf_counter() - t0)

            # ONNX 反归一化 + IFFT
            t0 = time.perf_counter()
            onnx_physical = onnx_freq_out[0, ..., 0] * std + mean + 1j * (onnx_freq_out[0, ..., 1] * std + mean)
            onnx_freq_image = (np.fft.ifft2(onnx_physical).real * 256)[None, ..., None].astype(np.float32)
            times["onnx_ifft"].append(time.perf_counter() - t0)

            # ONNX spatial 推理
            t0 = time.perf_counter()
            onnx_pred = run_onnx(onnx_spatial_sess, onnx_freq_image)[0, ..., 0]
            times["onnx_spatial"].append(time.perf_counter() - t0)

            times["onnx_total"].append(time.perf_counter() - t_onnx_total_start)

            # 精度指标
            fm = compute_metrics(predicted_kspace, onnx_freq_out)
            acc_metrics["freq_max_abs"].append(fm["max_abs"])
            acc_metrics["freq_rmse"].append(fm["rmse"])
            acc_metrics["freq_cos_sim"].append(fm["cos_sim"])

            psnr = compute_psnr(prediction, onnx_pred)
            mse = float(np.mean((prediction - onnx_pred) ** 2))
            acc_metrics["img_psnr"].append(psnr)
            acc_metrics["img_mse"].append(mse)
        # ===================================================================

    # ========== 输出计时报告 ==========
    print("\n" + "=" * 60)
    print(f"Pipeline Benchmark Report (precision={args.precision})")
    print("=" * 60)
    n = len(times["total"])
    for key in ["preprocess", "frequency", "ifft", "spatial", "postprocess", "total"]:
        arr = np.array(times[key]) * 1000
        print(f"{key:14s}: mean={arr.mean():7.2f} ms | "
              f"min={arr.min():7.2f} | max={arr.max():7.2f} | "
              f"P50={np.percentile(arr, 50):7.2f} | P99={np.percentile(arr, 99):7.2f}")

    # 如果启用了 ONNX 基准，追加 ONNX 计时
    if times["onnx_total"]:
        print("-" * 60)
        for key in ["onnx_frequency", "onnx_ifft", "onnx_spatial", "onnx_total"]:
            arr = np.array(times[key]) * 1000
            print(f"{key:14s}: mean={arr.mean():7.2f} ms | "
                  f"min={arr.min():7.2f} | max={arr.max():7.2f} | "
                  f"P50={np.percentile(arr, 50):7.2f} | P99={np.percentile(arr, 99):7.2f}")

    print("=" * 60)
    print(f"Throughput (TRT): {1 / np.mean(times['total']):.2f} FPS")
    if times["onnx_total"]:
        print(f"Throughput (ONNX): {1/ np.mean(times['onnx_total']):.2f} FPS")
        speedup = np.mean(times["onnx_total"]) / np.mean(times["total"])
        print(f"Speedup (ONNX / TRT_{args.precision}): {speedup:.2f}x")
    print(f"Samples: {n}")
    print("=" * 60)
    # ==================================

    # ========== 输出精度报告 ==========
    if acc_metrics["freq_max_abs"]:
        print("\n" + "=" * 60)
        print(f"Accuracy Report vs ONNX baseline (precision={args.precision})")
        print("=" * 60)
        print(f"Frequency output:")
        print(f"  Max Abs Error : {np.mean(acc_metrics['freq_max_abs']):.6f}")
        print(f"  RMSE          : {np.mean(acc_metrics['freq_rmse']):.6f}")
        print(f"  Cosine Sim    : {np.mean(acc_metrics['freq_cos_sim']):.6f}")
        print(f"Final image:")
        print(f"  PSNR (dB)     : {np.mean(acc_metrics['img_psnr']):.4f}")
        print(f"  MSE           : {np.mean(acc_metrics['img_mse']):.6f}")
        print("=" * 60)
    # ==================================

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
