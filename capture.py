import cv2
import numpy as np
import csv
import serial
import re
import os

# =====================
# 保存フォルダ
# =====================

base_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\data"

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
cap.set(cv2.CAP_PROP_EXPOSURE, -4)

# =====================
# モード管理
# phase = "TILT" → 縦線（|）を使う
# phase = "PAN"  → 横線（===）を使う
# =====================

phase = "TILT"  # 初期値

# =====================
# マスクから輪郭を取得し
# モードに合った輪郭だけ残す
# =====================

def filter_contours_by_phase(mask, phase):
    """
    maskの輪郭を面積順に並べて、
    TILT → 縦寄り（高さ > 幅）の輪郭を選択
    PAN  → 横寄り（幅 > 高さ）の輪郭を選択
    選んだ輪郭だけを白く塗ったmaskを返す
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.zeros_like(mask)

    # 面積でソート（大きい順）、上位5つまで対象
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    filtered_mask = np.zeros_like(mask)

    for cnt in contours:

        if cv2.contourArea(cnt) < 50:  # 小さすぎるノイズは無視
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if phase == "TILT":
            # 縦線：高さが幅より大きい
            if h > w:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

        elif phase == "PAN":
            # 横線：幅が高さより大きい
            if w > h:
                cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    return filtered_mask


# =====================
# マスクからレーザー中心座標を取得
# =====================

def extract_points(mask, phase):
    """
    TILT（縦線）→ 各x列でyの平均を取る（従来通り）
    PAN（横線） → 各y行でxの平均を取る
    戻り値は常に (x, y) のリスト
    """

    points = []
    h, w = mask.shape

    if phase == "TILT":
        # 縦線：x列ごとにyを取得
        for x in range(w):
            ys = np.where(mask[:, x] > 0)[0]
            if len(ys) > 0:
                y = int(np.mean(ys))
                points.append((x, y))

    elif phase == "PAN":
        # 横線：y行ごとにxを取得
        for y in range(h):
            xs = np.where(mask[y, :] > 0)[0]
            if len(xs) > 0:
                x = int(np.mean(xs))
                points.append((x, y))

    return points


# =====================
# CSV保存
# =====================

def save_csv(points, phase, tilt, pan):
    """
    pointsを (x, y, dy, groove) 形式でCSV保存
    dy と groove は隣接点間の変化量から計算
    """

    filename = os.path.join(
        base_dir,
        f"T{tilt:03d}_P{pan:03d}.csv"
    )

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "dy", "groove"])

        for i in range(len(points)):
            dy = 0
            if i > 0:
                dy = abs(points[i][1] - points[i - 1][1])
            groove = 1 if dy > 5 else 0
            writer.writerow([points[i][0], points[i][1], dy, groove])

    print(f"保存完了: {filename}  ({len(points)}点)")


# =====================
# メインループ
# =====================

print("起動しました。qで終了。")
print(f"初期フェーズ: {phase}")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # --- HSV変換・赤色抽出 ---

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0,   20, 29])
    upper1 = np.array([15, 255, 255])
    lower2 = np.array([160, 20,  20])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask  = cv2.bitwise_or(mask1, mask2)

    # ノイズ除去
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    # --- フェーズに合った輪郭だけ残す ---

    filtered_mask = mask    # filter_contours_by_phase(mask, phase)

    # --- 座標取得 ---

    points = extract_points(filtered_mask, phase)

    # --- 描画 ---

    display = frame.copy()

    for i in range(len(points) - 1):
        cv2.line(display, points[i], points[i + 1], (255, 0, 0), 1)

    # 現在のフェーズを画面に表示
    cv2.putText(
        display,
        f"Phase: {phase}  Points: {len(points)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("result", display)
    cv2.imshow("mask_raw", mask)
    cv2.imshow("mask_filtered", filtered_mask)

    # --- Arduino受信 ---

    if ser.in_waiting:

        msg = ser.readline().decode(errors="ignore").strip()
        print(msg)

        # フェーズ切り替え
        if "PHASE1: TILT SCAN" in msg:
            phase = "TILT"
            print(f"→ フェーズ切り替え: TILT（縦線モード）")

        elif "PHASE2: PAN SCAN" in msg:
            phase = "PAN"
            print(f"→ フェーズ切り替え: PAN（横線モード）")

        # 撮影トリガー
        m = re.search(r"SHOT:T(\d+):P(\d+)", msg)

        if m:
            tilt = int(m.group(1))
            pan  = int(m.group(2))

            if len(points) > 0:
                save_csv(points, phase, tilt, pan)
            else:
                print(f"警告: points が空です → {msg} はスキップ")

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
ser.close()
cv2.destroyAllWindows()