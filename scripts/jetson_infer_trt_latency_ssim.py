"""Run the two-engine TensorRT dual-domain pipeline on Jetson.

Requires the JetPack TensorRT Python API and pycuda.  IFFT is intentionally
kept in FP32 outside TensorRT to preserve the model's training contract.

Extended with:
  - End-to-end timing with / without I/O
  - Optional FP32 TensorRT baseline accuracy comparison
  - Warmup (10 images, covers both main and baseline engines)
  - Image preloading, async PNG save (compress_level=1)
  - Two speedup metrics: engines-only and with-preprocess
  - GT-based PSNR/SSIM/RMSE evaluation against reference images
  - Packed .npy output and metrics CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


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
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup images (default: 10)")
    parser.add_argument("--reference-dir", type=Path, default=None,
                        help="Directory of ground-truth images (default: <input_parent>/reference)")
    parser.add_argument("--save-png", action="store_true",
                        help="Save reconstruction PNGs (default: False)")
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


def extract_base_name(path: Path) -> str:
    """从输入文件名提取 base name。
    输入格式: img_29997_SR10.5_RMSE0.0269_SSIM0.8603.png
    输出: img_29997
    """
    stem = path.stem
    match = re.match(r"^(.+?)_SR[\d.]+_RMSE[\d.]+_SSIM[\d.]+$", stem)
    if match:
        return match.group(1)
    # 兜底：尝试去掉最后一个下划线之后的内容
    return stem


def find_gt_path(base: str, ref_dir: Path) -> Path | None:
    """查找对应的 GT 文件。"""
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        for pattern in (f"{base}_original{ext}", f"{base}{ext}"):
            p = ref_dir / pattern
            if p.exists():
                return p
    return None


def load_gt(path: Path) -> np.ndarray:
    """加载 GT 图像并归一化到 [0,1]。"""
    with Image.open(path) as image:
        image = image.convert("L")
        if image.size != (256, 256):
            image = image.resize((256, 256))
        return np.asarray(image, dtype=np.float32) / 255.0


def compute_psnr_ssim_rmse(pred: np.ndarray, gt: np.ndarray) -> dict:
    """计算重建图像与 GT 的 PSNR/SSIM/RMSE。"""
    mse = float(np.mean((pred - gt) ** 2))
    rmse = float(np.sqrt(mse))
    psnr = float('inf') if mse == 0 else float(10 * np.log10(1.0 / mse))
    ssim_val = float('nan')
    if _HAS_SKIMAGE:
        ssim_val = float(_ssim(gt, pred, data_range=1.0))
    return {"PSNR": psnr, "SSIM": ssim_val, "RMSE": rmse}


def compute_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    """原有的 vs FP32 基准的频域指标。"""
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

    if not _HAS_SKIMAGE:
        print("[WARN] scikit-image not installed. SSIM will be NaN. Install: pip3 install --user scikit-image")

    # ========== GT 目录 ==========
    if args.reference_dir is not None:
        ref_dir = args.reference_dir.resolve()
    else:
        # 默认从 input 的父目录下的 reference 找
        ref_dir = args.input.resolve().parent / "reference"
    print(f"[INFO] Reference dir: {ref_dir} (exists: {ref_dir.exists()})")

    # ========== FP32 基准引擎初始化（可选） ==========
    baseline_freq = None
    baseline_spatial = None
    if args.baseline_fp32 and args.precision != "fp32":
        baseline_freq = EngineRunner(engines / "frequency_fp32.engine")
        baseline_spatial = EngineRunner(engines / "spatial_fp32.engine")
        print("[INFO] FP32 baseline engines loaded.")
    elif args.baseline_fp32 and args.precision == "fp32":
        print("[WARN] precision=fp32, baseline comparison skipped (same as target).")

    # ========== 图片预加载 ==========
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

    # ========== 预热 N 张图 ==========
    warmup_n = min(args.warmup, len(image_cache))
    print(f"[INFO] Warming up with {warmup_n} images (main + baseline engines)...")
    for idx in range(warmup_n):
        _, source = image_cache[idx]
        spectrum = np.fft.fft2(source) / fft_scale
        normalized = (np.stack((spectrum.real, spectrum.imag), axis=-1)[None].astype(np.float32) - mean) / std

        pk = frequency(normalized)
        phy = pk[0, ..., 0] * std + mean + 1j * (pk[0, ..., 1] * std + mean)
        fi = (np.fft.ifft2(phy).real * 256)[None, ..., None].astype(np.float32)
        _ = spatial(fi)

        if baseline_freq is not None and baseline_spatial is not None:
            base_pk = baseline_freq(normalized)
            base_phy = base_pk[0, ..., 0] * std + mean + 1j * (base_pk[0, ..., 1] * std + mean)
            base_fi = (np.fft.ifft2(base_phy).real * 256)[None, ..., None].astype(np.float32)
            _ = baseline_spatial(base_fi)
    print("[INFO] Warmup done.")

    # ========== 计时容器 ==========
    times = {
        "read_image": read_times,
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

    # ========== 收集重建结果和 GT 指标 ==========
    reconstructions = []          # 每个元素 shape (256, 256)，用于打包 npy
    reconstruction_names = []     # 对应的 base name
    gt_metrics_rows = []          # CSV 行

    # ========== 异步 PNG 保存 ==========
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
        base_name = extract_base_name(source_path)
        gt_path = find_gt_path(base_name, ref_dir) if ref_dir.exists() else None
        gt = load_gt(gt_path) if gt_path is not None else None

        # 1. 预处理
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

        # 5. 计算 GT 指标
        if gt is not None:
            gt_metric = compute_psnr_ssim_rmse(prediction, gt)
            psnr_val = gt_metric["PSNR"]
            ssim_val = gt_metric["SSIM"]
            rmse_val = gt_metric["RMSE"]
            gt_metrics_rows.append({
                "filename": source_path.name,
                "precision": args.precision,
                "PSNR": f"{psnr_val:.4f}",
                "SSIM": f"{ssim_val:.4f}",
                "RMSE": f"{rmse_val:.4f}",
            })
        else:
            psnr_val, ssim_val, rmse_val = float('nan'), float('nan'), float('nan')
            gt_metrics_rows.append({
                "filename": source_path.name,
                "precision": args.precision,
                "PSNR": "NaN",
                "SSIM": "NaN",
                "RMSE": "NaN",
            })

        # 6. 收集重建结果
        reconstructions.append(prediction.astype(np.float32))
        reconstruction_names.append(base_name)

        # 7. 可选保存 PNG（带实际指标）
        t0 = time.perf_counter()
        if args.save_png:
            if not np.isnan(rmse_val) and not np.isnan(ssim_val):
                png_name = f"{base_name}_RMSE{rmse_val:.4f}_SSIM{ssim_val:.4f}.png"
            else:
                png_name = f"{base_name}.png"
            executor.submit(
                save_png_async,
                prediction,
                args.output_dir / png_name,
                idx,
            )

        times["save_submit"].append(time.perf_counter() - t0)
        # 8. FP32 基准对比（可选）
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

    # 等待异步保存完成
    executor.shutdown(wait=True)
    times["save_actual"] = [t for t in save_actual_by_index if t is not None]

    # ========== 保存打包 npy ==========
    npy_path = args.output_dir / f"reconstructions_{args.precision}.npy"
    if reconstructions:
        np.save(npy_path, np.stack(reconstructions, axis=0))
        print(f"[INFO] Saved packed reconstructions to {npy_path} (shape: {np.stack(reconstructions, axis=0).shape})")

    # ========== 保存 CSV ==========
    csv_path = args.output_dir / f"metrics_{args.precision}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "precision", "PSNR", "SSIM", "RMSE"])
        writer.writeheader()
        writer.writerows(gt_metrics_rows)
    print(f"[INFO] Saved metrics CSV to {csv_path}")

    # ========== 三组总时间 ==========
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

    # ========== 加速比：三个口径 ==========
    if times["base_frequency"]:
        pre_arr = np.array(times["preprocess"])
        freq_arr = np.array(times["frequency"])
        ift_arr = np.array(times["ifft"])
        spa_arr = np.array(times["spatial"])
        base_freq_arr = np.array(times["base_frequency"])
        base_ifft_arr = np.array(times["base_ifft"])
        base_spa_arr = np.array(times["base_spatial"])

        speedup_two = np.mean(base_freq_arr + base_spa_arr) / np.mean(freq_arr + spa_arr)
        speedup_engine = np.mean(base_freq_arr + base_ifft_arr + base_spa_arr) / np.mean(freq_arr + ift_arr + spa_arr)
        speedup_full = np.mean(pre_arr + base_freq_arr + base_ifft_arr + base_spa_arr) / np.mean(pre_arr + freq_arr + ift_arr + spa_arr)

        print(f"Speedup (two engines only)   : {speedup_two:.2f}x")
        print(f"Speedup (engines + IFFT)     : {speedup_engine:.2f}x")
        print(f"Speedup (with preprocess)    : {speedup_full:.2f}x")
        print("=" * 60)

    # ========== vs FP32 精度报告 ==========
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

    # ========== vs GT 精度报告 ==========
    valid_gt = [r for r in gt_metrics_rows if r["PSNR"] != "NaN"]
    if valid_gt:
        psnrs = [float(r["PSNR"]) for r in valid_gt]
        ssims = [float(r["SSIM"]) for r in valid_gt if r["SSIM"] != "NaN"]
        rmses = [float(r["RMSE"]) for r in valid_gt]
        print("\n" + "=" * 60)
        print(f"GT-based Accuracy Report (precision={args.precision})")
        print("=" * 60)
        print(f"Samples with GT: {len(valid_gt)} / {len(gt_metrics_rows)}")
        print(f"PSNR: mean={np.mean(psnrs):.4f} dB | min={np.min(psnrs):.4f} | max={np.max(psnrs):.4f}")
        if ssims:
            print(f"SSIM: mean={np.mean(ssims):.4f} | min={np.min(ssims):.4f} | max={np.max(ssims):.4f}")
        print(f"RMSE: mean={np.mean(rmses):.4f} | min={np.min(rmses):.4f} | max={np.max(rmses):.4f}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
