import cv2
import numpy as np
import os
import re
from PIL import Image as PILImage

# =====================
# パス設定
# =====================

image_dir  = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\image"
result_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\result"

os.makedirs(result_dir, exist_ok=True)

threshold = 12  # 溝判定の深さ閾値（xのズレ）

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

    xs = np.array([p[0] for p in points])
    x_median = np.median(xs)
    x_std = np.std(xs)
    points = [(x, y) for x, y in points if abs(x - x_median) < x_std * 1]

    return points

# =====================
# 全画像処理
# =====================

all_points    = []
groove_points = []
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

        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)

        # 縦線用baseline：yを基準にxをfitting
        coef     = np.polyfit(ys, xs, 1)
        baseline = np.polyval(coef, ys)
        depth    = xs - baseline  # xのズレが溝の深さ

        for xi, yi in zip(xs, ys):
            xi, yi = int(xi), int(yi)
            if 0 <= xi < w and 0 <= yi < h:
                all_points.append((xi, yi))

        for i in range(1, len(points)):
            dx = abs(points[i][0] - points[i-1][0])
            if dx > threshold:
                px, py = points[i]
                if 0 <= px < w and 0 <= py < h:
                    groove_points.append((px, py))
        processed += 1

    except Exception as e:
        print(f"エラー: {fname} → {e}")
        skipped += 1

print(f"\n処理完了: {processed}枚, スキップ: {skipped}枚")
print(f"全レーザー点: {len(all_points)}点")
print(f"溝判定点:     {len(groove_points)}点")

# =====================
# 描画
# =====================

# 全レーザー点を白で描画
for px, py in all_points:
    cv2.circle(canvas, (px, py), 2, (255, 255, 255), -1)

# 溝点を緑で上書き
for px, py in groove_points:
    cv2.circle(canvas, (px, py), 3, (0, 220, 80), -1)

# =====================
# 保存
# =====================

out_path = os.path.join(result_dir, "2sole_map.png")
PILImage.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(out_path)

print(f"\n保存完了: {out_path}")