import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image as PILImage

# =====================
# パス設定
# =====================

image_dir  = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\image"
points_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\points"
groove_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\groove"
result_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\result"

os.makedirs(result_dir, exist_ok=True)

# =====================
# 背景画像の準備
# =====================

bg_path = os.path.join(image_dir, "background.png")

if os.path.exists(bg_path):
    canvas = np.array(PILImage.open(bg_path))
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    print("背景画像読み込み完了")
else:
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    print("背景画像なし → 黒背景")

h, w = canvas.shape[:2]

# =====================
# 全レーザー点（白）を描画
# =====================

total_points = 0
for fname in sorted(os.listdir(points_dir)):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(points_dir, fname))
    for x, y in zip(df["x"].astype(int), df["y"].astype(int)):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (x, y), 2, (255, 255, 255), -1)
            total_points += 1

# =====================
# 溝点（緑）を上書き描画
# =====================

total_groove = 0
for fname in sorted(os.listdir(groove_dir)):
    if not fname.endswith(".csv"):
        continue
    df = pd.read_csv(os.path.join(groove_dir, fname))
    for x, y in zip(df["x"].astype(int), df["y"].astype(int)):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(canvas, (x, y), 3, (0, 220, 80), -1)
            total_groove += 1

print(f"全レーザー点: {total_points}点")
print(f"溝判定点:     {total_groove}点")

# =====================
# 保存
# =====================

out_path = os.path.join(result_dir, "4sole_map.png")
PILImage.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(out_path)

print(f"\n保存完了: {out_path}")