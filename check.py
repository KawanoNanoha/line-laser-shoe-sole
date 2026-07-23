import cv2
import numpy as np
from PIL import Image as PILImage

# =====================
# 確認したい画像を指定
# =====================

fpath     = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\image\T100_P130.png"
result_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\result"

# =====================
# 画像読み込み
# =====================

img = cv2.cvtColor(np.array(PILImage.open(fpath)), cv2.COLOR_RGB2BGR)
img_bright = cv2.convertScaleAbs(img, alpha=5.0, beta=30)

# =====================
# ステップ1：赤色抽出
# =====================

hsv = cv2.cvtColor(img_bright, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(hsv, np.array([0,   80, 80]), np.array([15,  255, 255]))
mask2 = cv2.inRange(hsv, np.array([160, 80, 80]), np.array([180, 255, 255]))
mask  = cv2.bitwise_or(mask1, mask2)
kernel = np.ones((3,3), np.uint8)
mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# 元画像に青でオーバーレイ
step1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
step1[mask > 0] = [0, 0, 255]

PILImage.fromarray(step1).save(f"{result_dir}\\step1_red.png")
print(f"Step1保存: step1_red.png  ({mask.sum()//255}px)")

# =====================
# ステップ2：横線排除
# =====================

# HoughLinesで横線を除去
lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=10)
hough_mask = np.zeros_like(mask)
if lines is not None:
    for l in lines:
        x1,y1,x2,y2 = l[0]
        angle = np.degrees(np.arctan2(y2-y1, x2-x1))
        if not (-30 < angle < 30):
            cv2.line(hough_mask, (x1,y1), (x2,y2), 255, 10)
step2a = cv2.bitwise_and(mask, hough_mask)

# さらに輪郭の縦横比でフィルター
contours, _ = cv2.findContours(step2a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_mask = np.zeros_like(mask)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if h > w * 1.0:
        cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

step2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
step2[filtered_mask > 0] = [0, 0, 255]

PILImage.fromarray(step2).save(f"{result_dir}\\step2_vertical.png")
print(f"Step2保存: step2_vertical.png  ({filtered_mask.sum()//255}px)")

# =====================
# ステップ3：溝検出
# =====================

red = img_bright[:,:,2]
points = []
for y in range(filtered_mask.shape[0]):
    xs = np.where(filtered_mask[y,:] > 0)[0]
    if len(xs) == 0:
        continue
    x = xs[np.argmax(red[y, xs])]
    points.append((int(x), int(y)))

print(f"縦線検出点数: {len(points)}")

threshold = 8
groove_pts = []
for i in range(1, len(points)):
    dx = abs(points[i][0] - points[i-1][0])
    if dx > threshold:
        groove_pts.append(points[i])

step3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
step3[filtered_mask > 0] = [0, 0, 255]  # 縦線を青で
for px, py in groove_pts:
    cv2.circle(step3, (px, py), 6, (0, 255, 0), -1)  # 溝を緑で

PILImage.fromarray(step3).save(f"{result_dir}\\step3_groove.png")
print(f"Step3保存: step3_groove.png  (溝{len(groove_pts)}点)")