import pandas as pd
import numpy as np
import cv2
import os
import re

# =====================
# パス設定
# =====================

data_dir   = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\data"
result_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\result"

os.makedirs(result_dir, exist_ok=True)

# =====================
# 背景画像の設定
# 靴底の写真があればそのパスを指定
# なければ黒背景になる
# =====================

background_path = None
# 例: background_path = r"C:\Users\siro1\k22057nk\研究\卒業研究\shoe_background.jpg"

threshold = 10

# =====================
# 背景画像の準備
# =====================

if background_path and os.path.exists(background_path):
    canvas = cv2.imread(background_path)
    h, w = canvas.shape[:2]
    print(f"背景画像読み込み: {w}x{h}")
else:
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    h, w = 480, 640
    print("背景なし（黒背景）")

# =====================
# CSVから溝点を収集
# =====================

all_points    = []  # 全レーザー点
groove_points = []  # 溝判定点

for fname in sorted(os.listdir(data_dir)):
    if not fname.endswith(".csv"):
        continue

    m = re.match(r"T(\d+)_P(\d+)\.csv", fname)
    if not m:
        continue

    tilt_angle = int(m.group(1))
    pan_angle  = int(m.group(2))
    # # sole_mapping_fixed.py の tilt_angle フィルターを一時的に追加
    # if tilt_angle != 50:
    #     continue

    # Tiltスキャン（Pan=080固定）のみ使う
    if pan_angle != 130:
        continue

    fpath = os.path.join(data_dir, fname)

    try:
        df = pd.read_csv(fpath)
        if df.empty or "x" not in df.columns or "y" not in df.columns:
            continue
        if len(df) < 3:
            continue

        x = df["x"].values
        y = df["y"].values

        # # 中央値から外れた点を除外
        # y_median = np.median(y)
        # y_std = np.std(y)
        # mask_filter = np.abs(y - y_median) < y_std * 1.5
        # x = x[mask_filter]
        # y = y[mask_filter]

        # # x範囲が広すぎる断面はスキップ（横線混入）
        # if x.max() - x.min() > 250:
        #     continue

        # baseline基準で深さ再計算
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
        print(f"スキップ: {fname} → {e}")

print(f"全レーザー点: {len(all_points)}点")
print(f"溝判定点:     {len(groove_points)}点")

# =====================
# 描画
# =====================

# 全レーザー点（暗い青）
# for px, py in all_points:
#     cv2.circle(canvas, (px, py), 2, (80, 80, 160), -1)

# 全レーザー点を白で描画（溝判定なし）
for px, py in all_points:
    cv2.circle(canvas, (px, py), 2, (255, 255, 255), -1)

# 溝点（緑）
# for px, py in groove_points:
#     cv2.circle(canvas, (px, py), 4, (0, 220, 80), -1)

# =====================
# 保存
# =====================

from PIL import Image
out_path = os.path.join(result_dir, "sole_mapping.png")
canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
Image.fromarray(canvas_rgb).save(out_path)
# cv2.imwrite(out_path, canvas)

print(f"\n保存完了: {out_path}")