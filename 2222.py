import cv2
import numpy as np
import csv

cap = cv2.VideoCapture(0)

# 露出設定
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)

def mouse_click(event, x, y, flags, param):
    global hsv

    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"HSV at ({x},{y}) : {hsv[y,x]}")

cv2.namedWindow("result")
cv2.setMouseCallback("result", mouse_click)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 赤色抽出
    lower1 = np.array([0, 20, 29])
    upper1 = np.array([15, 255, 255])

    lower2 = np.array([160, 20, 20])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)

    # ノイズ除去
    kernel = np.ones((2,2), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # 平滑化
    mask = cv2.GaussianBlur(
        mask,
        (5,5),
        0
    )

    # 再二値化
    _, mask = cv2.threshold(
        mask,
        100,
        255,
        cv2.THRESH_BINARY
    )

    # 少し太らせる
    # mask = cv2.dilate(mask, kernel, iterations=1)

    display = frame.copy()

    # レーザー部分を緑色表示
    display[mask > 0] = [0,255,0]
    points = []
    h, w = mask.shape

    # レーザー中心取得
    for x in range(w):

        ys = np.where(mask[:, x] > 0)[0]

        if len(ys) > 0:
            y = int(np.mean(ys))
            points.append((x, y))

    # 中心線を描画
    for i in range(len(points)-1):
        cv2.line(
        display,
        points[i],
        points[i+1],
        (255,0,0),
        1
    )

    cv2.imshow("result", display)
    cv2.imshow("mask", mask)
    cv2.imshow("original", frame)

    key = cv2.waitKey(1)


    # Sキーで保存
    if key == ord('s'):

        with open("laser.csv", "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(["x", "y", "dy", "groove"])

            for i in range(len(points)):

                dy = 0

                if i > 0:
                    dy = abs(points[i][1] - points[i-1][1])

                groove = 0

                if dy > 5:
                    groove = 1

                writer.writerow([
                    points[i][0],
                    points[i][1],
                    dy,
                    groove
                ])

        print("laser.csv 保存完了")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()