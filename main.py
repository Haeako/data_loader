import time
import torch
import pyspng              # pip install pyspng

# 1. Đọc toàn bộ file PNG dưới dạng bytes
with open("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/"
          "Track4_AIC_FishEyecamera/datasets/fisheye1k/"
          "camera19_A_1.png", "rb") as fin:
    img_bytes = fin.read()

# 2. Đo thời gian decode + tensor conversion
t0 = time.perf_counter()

# 2.1 Decode bytes → numpy array (H, W, 3), RGB8
# pyspng.load() hoặc pyspng.load_png()
try:
    image_data = pyspng.load(img_bytes)    # :contentReference[oaicite:1]{index=1}
except AttributeError:
    image_data = pyspng.load_png(img_bytes)  # :contentReference[oaicite:2]{index=2}

# 2.2 Chuyển NumPy → PyTorch tensor [1,3,H,W], float32, normalize
tensor = (
    torch.from_numpy(image_data)      # (H, W, 3)
         .permute(2, 0, 1)            # (3, H, W)
         .unsqueeze(0)                # (1, 3, H, W)
         .to(torch.float32)           # float32
         .div_(255.0)                 # normalize [0,1]
)

t1 = time.perf_counter()

# 3. Kết quả
print(f"Decoded shape: {image_data.shape}")
print(f"Tensor shape: {tensor.shape}, dtype={tensor.dtype}")
print(f"Total decode → tensor time: {t1 - t0:.4f}s")
