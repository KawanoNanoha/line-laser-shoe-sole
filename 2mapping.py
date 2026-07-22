import pandas as pd
import numpy as np
import cv2
import os
import re
from PIL import Image

# =====================
# パス設定
# =====================

depth_dir  = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\depth"
result_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\result"

os.makedirs(result_dir, exist_ok=True)

# =====================
# キャンバス
# =====================

canvas_groove = np.zeros((480, 640, 3), dtype=np.uint8)
canvas_depth  = np.zeros((480, 640, 3), dtype=np.uint8)

# =====================
# depth CSVを全部読み込んでマッピング
# =====================

all_points    = []  # (x, y, depth)
groove_points = []  # (x, y, depth)

processed = 0
skipped   = 0

for fname in sorted(os.listdir(depth_dir)):
    if not fname.endswith(".csv"):
        continue

    m = re.match(r"T(\d+)_P(\d+)\.csv", fname)
    if not m:
        continue

    tilt_angle = int(m.group(1))
    pan_angle  = int(m.group(2))

    # # Tiltスキャン（Pan=130固定）のみ使う
    # if pan_angle != 130:
    #     continue

    # 異常値スキップ
    if tilt_angle > 1500:
        continue

    fpath = os.path.join(depth_dir, fname)

    try:
        df = pd.read_csv(fpath)

        if len(df) < 3:
            skipped += 1
            continue

        x      = df["x"].values.astype(float)
        y      = df["y"].values.astype(float)
        depth  = df["depth"].values.astype(float)
        groove = df["groove"].values.astype(int)

        for xi, yi, di, gi in zip(x, y, depth, groove):
            xi, yi = int(xi), int(yi)
            if 0 <= xi < 640 and 0 <= yi < 480:
                all_points.append((xi, yi, di))
                if gi == 1:
                    groove_points.append((xi, yi, di))

        processed += 1

    except Exception as e:
        print(f"エラー: {fname} → {e}")
        skipped += 1

print(f"処理完了: {processed}枚, スキップ: {skipped}枚")
print(f"全レーザー点: {len(all_points)}点")
print(f"溝判定点:     {len(groove_points)}点")

# =====================
# 描画
# =====================

# 深さマップ
max_depth = max([d for _, _, d in all_points], default=1)
max_depth = max(max_depth, 1)

for xi, yi, di in all_points:
    intensity = int(min(max(di, 0) / max_depth, 1.0) * 255)
    color = (0, intensity, intensity)  # 深いほど明るいシアン
    cv2.circle(canvas_depth, (xi, yi), 2, color, -1)

# 全レーザー点を白で描画
for xi, yi, di in all_points:
    cv2.circle(canvas_groove, (xi, yi), 2, (255, 255, 255), -1)

# 溝点を緑で上書き
for xi, yi, di in groove_points:
    cv2.circle(canvas_groove, (xi, yi), 3, (0, 220, 80), -1)

# =====================
# 保存
# =====================

out_groove = os.path.join(result_dir, "sole_groove_map.png")
out_depth  = os.path.join(result_dir, "sole_depth_map.png")

Image.fromarray(cv2.cvtColor(canvas_groove, cv2.COLOR_BGR2RGB)).save(out_groove)
Image.fromarray(cv2.cvtColor(canvas_depth,  cv2.COLOR_BGR2RGB)).save(out_depth)

print(f"\n保存完了:")
print(f"  {out_groove}")
print(f"  {out_depth}")