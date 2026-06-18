import cv2
import numpy as np
import csv

cap = cv2.VideoCapture(0)

# 露出を少し明るくする
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

    # 赤色抽出範囲を広げる
    lower1 = np.array([0, 20, 29])
    upper1 = np.array([15, 255, 255])

    lower2 = np.array([160, 20, 20])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)

    # ノイズ除去
    kernel = np.ones((3,3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # 少し太らせる
    mask = cv2.dilate(mask, kernel, iterations=1)

    display = frame.copy()

    points = []

    h, w = mask.shape

    for x in range(w):

        ys = np.where(mask[:, x] > 0)[0]

        if len(ys) > 0:

            # 上端と下端の中点
            y = int((ys[0] + ys[-1]) / 2)

            points.append((x, y))

            cv2.circle(
                display,
                (x, y),
                1,
                (255, 0, 0),
                -1
            )

    # 中央点
    if len(points) > 0:

        center = points[len(points)//2]

        cv2.circle(
            display,
            center,
            6,
            (0,255,255),
            -1
        )

        cv2.putText(
            display,
            str(center),
            (20,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,255),
            2
        )

    # レーザー部分を緑表示
    display[mask > 0] = [0,255,0]

    cv2.imshow("result", display)
    cv2.imshow("mask", mask)
    cv2.imshow("original", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):

        with open("laser.csv", "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(["x","y"])

            writer.writerows(points)

        print("保存完了")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()