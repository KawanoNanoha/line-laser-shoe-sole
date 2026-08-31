import cv2
import numpy as np
import pandas as pd
import os
import re
from PIL import Image as PILImage
 
# =====================
# パス設定
# =====================
 
image_dir  = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\image"
points_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\points"
 
os.makedirs(points_dir, exist_ok=True)
 
# =====================
# レーザー検出関数
# =====================
 
def detect_laser(img):
    """
    縦線のx座標をy行ごとに取得する
    赤さが最大のxを各y行で1点だけ取る
    """
    img_bright = cv2.convertScaleAbs(img, alpha=5.0, beta=30)
 
    hsv = cv2.cvtColor(img_bright, cv2.COLOR_BGR2HSV)
 
    mask1 = cv2.inRange(hsv, np.array([0,   80, 80]), np.array([15,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
    mask  = cv2.bitwise_or(mask1, mask2)
 
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
 
    red = img_bright[:, :, 2]  # 赤チャンネル
 
    points = []
    for y in range(mask.shape[0]):
        xs = np.where(mask[y, :] > 0)[0]
        if len(xs) == 0:
            continue
        # 赤さが最大のxを取得
        values = red[y, xs]
        x = xs[np.argmax(values)]
        points.append((int(x), int(y)))
 
    if len(points) < 10:
        return points
 
    # 外れ値除去（中央値±標準偏差）
    xs_arr = np.array([p[0] for p in points])
    x_median = np.median(xs_arr)
    x_std = np.std(xs_arr)
    points = [(x, y) for x, y in points if abs(x - x_median) < x_std * 1]
 
    return points
 
# =====================
# 全画像処理
# =====================
 
processed = 0
skipped   = 0
 
for fname in sorted(os.listdir(image_dir)):
    if not fname.endswith(".png"):
        continue
    if fname == "background.png":
        continue
 
    m = re.match(r"T(\d+)_P(\d+)\.png", fname)
    if not m:
        continue
 
    tilt_angle = int(m.group(1))
    pan_angle  = int(m.group(2))
 
    # 異常値スキップ
    if tilt_angle > 200:
        continue
 
    # Tiltスキャン（Pan=130固定）のみ使う
    if pan_angle != 130:
        continue
 
    fpath = os.path.join(image_dir, fname)
 
    try:
        img = cv2.cvtColor(np.array(PILImage.open(fpath)), cv2.COLOR_RGB2BGR)
 
        points = detect_laser(img)
        print(f"{fname}: {len(points)}点, x中央値={int(np.median([p[0] for p in points])) if points else 'なし'}")
 
        if len(points) < 100:
            print(f"スキップ（点数不足）: {fname} ({len(points)}点)")
            skipped += 1
            continue
 
        df = pd.DataFrame(points, columns=["x", "y"])
        out = os.path.join(points_dir, fname.replace(".png", ".csv"))
        df.to_csv(out, index=False)
        print(f"保存: {out}")
 
        processed += 1
 
    except Exception as e:
        print(f"エラー: {fname} → {e}")
        skipped += 1
 
print(f"\n処理完了: {processed}枚, スキップ: {skipped}枚")
 



















