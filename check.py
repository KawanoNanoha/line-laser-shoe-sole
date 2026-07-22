from PIL import Image as PILImage
import cv2
import numpy as np

fpath = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\image\T100_P130.png"

img = cv2.cvtColor(np.array(PILImage.open(fpath)), cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

mask1 = cv2.inRange(hsv, np.array([0, 20, 29]), np.array([15, 255, 255]))
mask2 = cv2.inRange(hsv, np.array([160, 20, 20]), np.array([180, 255, 255]))
mask = cv2.bitwise_or(mask1, mask2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
mask = cv2.GaussianBlur(mask, (5,5), 0)
_, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=20, minLineLength=20, maxLineGap=10)

filtered = np.zeros_like(mask)
if lines is not None:
    for l in lines:
        x1,y1,x2,y2 = l[0]
        angle = np.degrees(np.arctan2(y2-y1, x2-x1))
        if -45 < angle < 45:
            cv2.line(filtered, (x1,y1), (x2,y2), 255, 3)

pts = []
for x in range(filtered.shape[1]):
    ys_col = np.where(filtered[:, x] > 0)[0]
    if len(ys_col) > 0:
        pts.append((x, int(np.mean(ys_col))))

print(f"検出点数: {len(pts)}")

if len(pts) >= 3:
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    bl = np.polyval(np.polyfit(xs, ys, 1), xs)
    depth = bl - ys
    print(f"depth範囲: {depth.min():.1f} ~ {depth.max():.1f}")
    print(f"threshold(10)超え: {(depth > 10).sum()}点")
    print(f"y範囲: {ys.min():.0f} ~ {ys.max():.0f}")
    print(f"x範囲: {xs.min():.0f} ~ {xs.max():.0f}")