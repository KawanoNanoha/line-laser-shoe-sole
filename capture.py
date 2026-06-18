import cv2
import numpy as np
import csv
import serial
import re
import os

# =====================
# 保存フォルダ
# =====================

base_dir = r"C:\Users\siro1\k22057nk\研究\卒業研究\data"

os.makedirs(base_dir, exist_ok=True)

print("保存先")
print(base_dir)

# =====================
# シリアル通信
# =====================

ser = serial.Serial(
    "COM3",
    9600,
    timeout=1
)

# =====================
# カメラ
# =====================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    # 赤色抽出

    lower1 = np.array([0, 20, 29])
    upper1 = np.array([15, 255, 255])

    lower2 = np.array([160, 20, 20])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(
        hsv,
        lower1,
        upper1
    )

    mask2 = cv2.inRange(
        hsv,
        lower2,
        upper2
    )

    mask = cv2.bitwise_or(
        mask1,
        mask2
    )

    # ノイズ除去

    kernel = np.ones((2, 2), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.GaussianBlur(
        mask,
        (5, 5),
        0
    )

    _, mask = cv2.threshold(
        mask,
        100,
        255,
        cv2.THRESH_BINARY
    )

    # =====================
    # レーザー中心取得
    # =====================

    points = []

    h, w = mask.shape

    for x in range(w):

        ys = np.where(
            mask[:, x] > 0
        )[0]

        if len(ys) > 0:

            y = int(np.mean(ys))

            points.append(
                (x, y)
            )

    # =====================
    # 描画
    # =====================

    display = frame.copy()

    for i in range(len(points) - 1):

        cv2.line(
            display,
            points[i],
            points[i + 1],
            (255, 0, 0),
            1
        )

    cv2.imshow(
        "result",
        display
    )

    cv2.imshow(
        "mask",
        mask
    )

    # =====================
    # Arduino受信
    # =====================

    if ser.in_waiting:

        msg = ser.readline() \
            .decode(errors="ignore") \
            .strip()

        print(msg)

        m = re.search(
            r"SHOT:T(\d+):P(\d+)",
            msg
        )

        if m:

            tilt = int(m.group(1))
            pan = int(m.group(2))

            filename = os.path.join(
                base_dir,
                f"T{tilt:03d}_P{pan:03d}.csv"
            )

            with open(
                filename,
                "w",
                newline=""
            ) as f:

                writer = csv.writer(f)

                writer.writerow(
                    [
                        "x",
                        "y",
                        "dy",
                        "groove"
                    ]
                )

                for i in range(len(points)):

                    dy = 0

                    if i > 0:

                        dy = abs(
                            points[i][1]
                            -
                            points[i - 1][1]
                        )

                    groove = 1 if dy > 5 else 0

                    writer.writerow(
                        [
                            points[i][0],
                            points[i][1],
                            dy,
                            groove
                        ]
                    )

            print("保存完了")
            print(filename)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
ser.close()
cv2.destroyAllWindows()