import cv2
import numpy as np
import os
import re
from PIL import Image

# =====================
# パス設定
# =====================

image_dir  = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\image"
result_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\result"

os.makedirs(result_dir, exist_ok=True)

threshold = 10  # 溝判定の深さ閾値

# =====================
# キャンバス
# =====================

canvas_groove = np.zeros((480, 640, 3), dtype=np.uint8)  # 溝マップ
canvas_depth  = np.zeros((480, 640, 3), dtype=np.uint8)  # 深さマップ

# =====================
# 画像からレーザー点を検出
# =====================

def detect_laser(img):
    """
    画像からレーザーの中心座標リストを返す
    横線（PANスキャン用）→ y行ごとにxの平均
    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, np.array([0,   20, 29]),  np.array([15,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 20, 20]),  np.array([180, 255, 255]))
    mask  = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    # # 行ごとの白ピクセル数で横線だけ残す
    # row_counts = np.sum(mask > 0, axis=1)
    # max_count  = row_counts.max()

    # if max_count == 0:
    #     return []

    
    lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=10)

    filtered_mask = np.zeros_like(mask)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line[0]
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            # 横線のみ残す（-45〜+45度）
            if -45 < angle < 45:
                cv2.line(filtered_mask, (x1,y1), (x2,y2), 255, 3)


    # y行ごとにxの平均を取得
    points = []
    for y in range(filtered_mask.shape[0]):
        xs = np.where(filtered_mask[y, :] > 0)[0]
        if len(xs) > 0:
            points.append((int(np.mean(xs)), y))

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

    m = re.match(r"T(\d+)_P(\d+)\.png", fname)
    if not m:
        continue

    tilt_angle = int(m.group(1))
    pan_angle  = int(m.group(2))

    # Tiltスキャン（Pan=130固定）のみ使う
    if pan_angle != 130:
        continue

    # # T1500などの異常値をスキップ
    # if tilt_angle > 200:
    #     continue

    fpath = os.path.join(image_dir, fname)

    try:
        from PIL import Image as PILImage
        pil_img = PILImage.open(fpath)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        points = detect_laser(img)

        if len(points) < 3:
            print(f"スキップ（点数不足）: {fname} ({len(points)}点)")
            skipped += 1
            continue

        # x, y に分解
        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)

        # x範囲が広すぎる場合はスキップ（横線ノイズ混入）
        if xs.max() - xs.min() > 250:
            print(f"スキップ（x範囲広すぎ）: {fname} (範囲={xs.max()-xs.min():.0f})")
            skipped += 1
            continue

        # baseline計算（xを基準にyをフィット）
        coef     = np.polyfit(xs, ys, 1)
        baseline = np.polyval(coef, xs)
        depth    = baseline - ys

        for xi, yi, di in zip(xs, ys, depth):
            xi, yi = int(xi), int(yi)
            if 0 <= xi < 640 and 0 <= yi < 480:
                all_points.append((xi, yi, di))
                if di > threshold:
                    groove_points.append((xi, yi, di))

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

# 深さの最大値（色スケール用）
max_depth = max([d for _, _, d in all_points], default=1)
max_depth = max(max_depth, 1)

for xi, yi, di in all_points:
    # 深さに応じて青→白のグラデーション
    intensity = int(min(di / max_depth, 1.0) * 200)
    color = (intensity, intensity, 80 + intensity)
    cv2.circle(canvas_depth, (xi, yi), 2, color, -1)

for xi, yi, di in all_points:
    if (xi, yi, di) in [(g[0], g[1], g[2]) for g in groove_points]:
        cv2.circle(canvas_groove, (xi, yi), 3, (0, 220, 80), -1)   # 緑=溝
    else:
        cv2.circle(canvas_groove, (xi, yi), 2, (80, 80, 160), -1)  # 青=平ら

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