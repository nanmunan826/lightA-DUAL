import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# 抽一张
input_path = "test_data/input/img_29349_SR9.7_RMSE0.0210_SSIM0.9175.JPEG"
gt_path = "test_data/reference/img_29349_original.JPEG"

inp = np.array(Image.open(input_path).convert("L"), dtype=np.float32) / 255.0
gt = np.array(Image.open(gt_path).convert("L"), dtype=np.float32) / 255.0

print("输入 vs GT SSIM:", ssim(gt, inp, data_range=1.0))
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# 1. 同一张图 vs 自己，SSIM 应该 = 1.0
img = np.array(Image.open("test_data/input/img_29349_SR9.7_RMSE0.0210_SSIM0.9175.JPEG").convert("L"), dtype=np.float32) / 255.0
print("自比 SSIM:", ssim(img, img, data_range=1.0))   # 应该是 1.0
print("自比 PSNR:", psnr(img, img, data_range=1.0))   # 应该是 inf

# 2. 加一点噪声，看指标是否合理下降
noisy = np.clip(img + np.random.normal(0, 0.05, img.shape), 0, 1)
print("加噪 SSIM:", ssim(img, noisy, data_range=1.0))  # 应该 < 1.0
print("加噪 PSNR:", psnr(img, noisy, data_range=1.0))  # 应该在 20~30 dB


from skimage.metrics import structural_similarity as ssim

img = np.array(Image.open("test_data/input/img_29349_SR9.7_RMSE0.0210_SSIM0.9175.JPEG").convert("L"), dtype=np.float32) / 255.0
gt = np.array(Image.open("test_data/reference/img_29349_original.JPEG").convert("L"), dtype=np.float32) / 255.0

# 默认参数（均匀窗口）
print("均匀窗口:", ssim(gt, img, data_range=1.0))

# 高斯窗口（接近原始论文）
print("高斯窗口:", ssim(gt, img, data_range=1.0, gaussian_weights=True, sigma=1.5, win_size=11))
