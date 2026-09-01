import cv2
import numpy as np
from PIL import Image as PILImage

# =====================
# パス設定
# =====================

fpath      = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\image\T114_P130.png"
result_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\result"

import os
os.makedirs(result_dir, exist_ok=True)

# 溝と判定する閾値
DEV_THRESHOLD      = 4    # 基準線からのズレ（px）がこれを超えたら溝
STRENGTH_THRESHOLD = 140  # 赤さの平均値がこれを下回ったら溝（赤が弱い＝反射不良）

# =====================
# 画像読み込み・赤色抽出
# =====================

img = cv2.cvtColor(np.array(PILImage.open(fpath)), cv2.COLOR_RGB2BGR)
img_bright = cv2.convertScaleAbs(img, alpha=2.0, beta=30)
hsv = cv2.cvtColor(img_bright, cv2.COLOR_BGR2HSV)

# 彩度の下限を緩め，白飛びした芯まで拾えるようにしている
mask1 = cv2.inRange(hsv, np.array([0,   30, 80]), np.array([15,  255, 255]))
mask2 = cv2.inRange(hsv, np.array([160, 30, 80]), np.array([180, 255, 255]))
mask  = cv2.bitwise_or(mask1, mask2)

kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

red = img_bright[:, :, 2]  # 赤チャンネル

# =====================
# ①横線排除
# =====================

lines = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=20, minLineLength=20, maxLineGap=10)
hough_mask = np.zeros_like(mask)
if lines is not None:
    for l in lines:
        x1, y1, x2, y2 = l[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if not (-30 < angle < 30):  # 水平に近い角度以外（＝縦線）を残す
            cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 10)
mask_vertical = cv2.bitwise_and(mask, hough_mask)

# 輪郭の縦横比で小さな孤立領域も除去
contours, _ = cv2.findContours(mask_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_mask = np.zeros_like(mask_vertical)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if h > w * 1.0:
        cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

step1_img = img.copy()
step1_img[filtered_mask > 0] = [0, 255, 0]
PILImage.fromarray(cv2.cvtColor(step1_img, cv2.COLOR_BGR2RGB)).save(
    os.path.join(result_dir, "step1_horizontal_removed.png"))
print("Step1保存: step1_horizontal_removed.png")

# =====================
# 行ごとの点群を取得（最も強いクラスタの重心）
# =====================

points = []
for y in range(filtered_mask.shape[0]):
    xs = np.where(filtered_mask[y, :] > 0)[0]
    if len(xs) == 0:
        continue
    values_all = red[y, xs].astype(np.float64)
    gaps = np.where(np.diff(xs) > 5)[0]
    cluster_bounds = np.split(np.arange(len(xs)), gaps + 1)
    best_idx, best_strength = None, -1
    for idx in cluster_bounds:
        s = values_all[idx].sum()
        if s > best_strength:
            best_strength = s
            best_idx = idx
    cxs = xs[best_idx]
    cvals = values_all[best_idx]
    x = int(round(np.sum(cxs * cvals) / np.sum(cvals)))
    strength = float(cvals.mean())  # その点の平均の赤さ
    points.append((x, y, strength))

print(f"検出点数: {len(points)}")

# =====================
# ②縦の基準線を作成
# =====================

ys       = np.array([p[1] for p in points], dtype=float)
xs_pts   = np.array([p[0] for p in points], dtype=float)
strengths = np.array([p[2] for p in points], dtype=float)

coef     = np.polyfit(ys, xs_pts, 1)   # y→xの1次直線フィッティング
baseline = np.polyval(coef, ys)

step2_img = img.copy()
for (x, y, s) in points:
    cv2.circle(step2_img, (x, int(y)), 1, (255, 255, 0), -1)
y_min, y_max = int(ys.min()), int(ys.max())
bx1, bx2 = int(np.polyval(coef, y_min)), int(np.polyval(coef, y_max))
cv2.line(step2_img, (bx1, y_min), (bx2, y_max), (0, 0, 255), 2)
PILImage.fromarray(cv2.cvtColor(step2_img, cv2.COLOR_BGR2RGB)).save(
    os.path.join(result_dir, "step2_baseline.png"))
print("Step2保存: step2_baseline.png")

# =====================
# ③溝の判定（基準線からのズレ or 赤の弱さ）
# =====================

deviations = np.abs(xs_pts - baseline)
groove_mask = (deviations > DEV_THRESHOLD) | (strengths < STRENGTH_THRESHOLD)

step3_img = img.copy()
for (x, y, s), is_groove in zip(points, groove_mask):
    color = (0, 220, 80) if is_groove else (255, 255, 0)  # 溝＝緑，通常＝水色
    cv2.circle(step3_img, (x, int(y)), 2, color, -1)
PILImage.fromarray(cv2.cvtColor(step3_img, cv2.COLOR_BGR2RGB)).save(
    os.path.join(result_dir, "step3_groove.png"))
print(f"Step3保存: step3_groove.png  （溝{int(groove_mask.sum())}点 / 全{len(points)}点）")

# =====================
# ④溝点のマッピング
# =====================

step4_img = np.zeros_like(img)
for (x, y, s), is_groove in zip(points, groove_mask):
    if is_groove:
        cv2.circle(step4_img, (x, int(y)), 3, (0, 220, 80), -1)
PILImage.fromarray(cv2.cvtColor(step4_img, cv2.COLOR_BGR2RGB)).save(
    os.path.join(result_dir, "step4_map.png"))
print("Step4保存: step4_map.png")