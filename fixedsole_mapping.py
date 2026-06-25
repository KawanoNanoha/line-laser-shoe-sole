import pandas as pd
import numpy as np
import cv2
import os
import re
from PIL import Image

# =====================
# パス設定
# =====================

data_dir   = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\data"
result_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\result"

os.makedirs(result_dir, exist_ok=True)

threshold = 10

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
h, w = 480, 640

# =====================
# CSVから溝点を収集
# =====================

all_points    = []
groove_points = []
skipped = 0

for fname in sorted(os.listdir(data_dir)):
    if not fname.endswith(".csv"):
        continue

    m = re.match(r"T(\d+)_P(\d+)\.csv", fname)
    if not m:
        continue

    tilt_angle = int(m.group(1))
    pan_angle  = int(m.group(2))

    # Tiltスキャン（Pan=130固定）のみ使う
    if pan_angle != 130:
        continue

    fpath = os.path.join(data_dir, fname)

    try:
        df = pd.read_csv(fpath)
        if df.empty or "x" not in df.columns or "y" not in df.columns:
            continue
        if len(df) < 3:
            continue

        x = df["x"].values.astype(float)
        y = df["y"].values.astype(float)

        # =====================
        # PANモードで取ったTiltスキャンCSVは
        # y=0,1,2,3...と連番になってる
        # → yを基準にfittingする（軸を入れ替え）
        # =====================

        y_range = y.max() - y.min()
        x_range = x.max() - x.min()

        if y_range > x_range:
            # yが広い範囲 → PANモードで取ったデータ
            # y基準でxをfitting
            coef     = np.polyfit(y, x, 1)
            baseline = np.polyval(coef, y)
            depth    = baseline - x  # xのズレが溝の深さ
        else:
            # x基準でyをfitting（通常のTiltモード）
            coef     = np.polyfit(x, y, 1)
            baseline = np.polyval(coef, x)
            depth    = baseline - y

        for xi, yi, di in zip(x, y, depth):
            xi, yi = int(xi), int(yi)
            if 0 <= xi < w and 0 <= yi < h:
                all_points.append((xi, yi))
                if di > threshold:
                    groove_points.append((xi, yi))

    except Exception as e:
        skipped += 1
        print(f"スキップ: {fname} → {e}")

print(f"全レーザー点: {len(all_points)}点")
print(f"溝判定点:     {len(groove_points)}点")
if skipped:
    print(f"スキップ:     {skipped}枚")

# =====================
# 描画
# =====================

for px, py in all_points:
    cv2.circle(canvas, (px, py), 2, (80, 80, 160), -1)

for px, py in groove_points:
    cv2.circle(canvas, (px, py), 4, (0, 220, 80), -1)

# =====================
# 保存（日本語パス対応）
# =====================

out_path = os.path.join(result_dir, "sole_mapping.png")
canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
Image.fromarray(canvas_rgb).save(out_path)

print(f"\n保存完了: {out_path}")