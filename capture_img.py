import cv2
import numpy as np
import serial
import re
import os

# =====================
# 保存フォルダ
# =====================

base_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\image"

os.makedirs(base_dir, exist_ok=True)

print("保存先:", base_dir)

# =====================
# シリアル通信
# =====================

ser = serial.Serial("COM3", 9600, timeout=1)

# =====================
# カメラ
# =====================

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, 0)

print(f"解像度: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print("起動しました。qで終了。")

saved_count = 0

# 最初のフレームを背景として保存
background_saved = False

while True:

    ret, frame = cap.read()

    # 最初の1枚を背景として保存
    if not background_saved:
        from PIL import Image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(os.path.join(base_dir, "background.png"))
        background_saved = True
        print("背景画像保存完了")

    if not ret:
        break

    # プレビュー表示
    display = frame.copy()
    cv2.putText(
        display,
        f"saved: {saved_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    cv2.imshow("preview", display)

    # Arduino受信
    if ser.in_waiting:
        msg = ser.readline().decode(errors="ignore").strip()
        print(msg)

        m = re.search(r"SHOT:T(\d+):P(\d+)", msg)
        if m:
            tilt = int(m.group(1))
            pan  = int(m.group(2))

            filename = os.path.join(
                base_dir,
                f"T{tilt:03d}_P{pan:03d}.png"
            )

            # 日本語パス対応で保存
            from PIL import Image
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_rgb).save(filename)

            saved_count += 1
            print(f"保存完了: {filename}")

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
ser.close()
cv2.destroyAllWindows()

print(f"\n終了。合計{saved_count}枚保存。")