import cv2
import numpy as np
import os
import re
from PIL import Image as PILImage


# =========================
# パス設定
# =========================

image_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\image"

result_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\red_mask"

os.makedirs(result_dir, exist_ok=True)


# =========================
# 赤色マスク作成
# =========================

def create_red_mask(img):

    # -------------------------
    # 明るさを少し補正
    # -------------------------

    img_bright = cv2.convertScaleAbs(
        img,
        alpha=1.0,
        beta=0
    )

    # -------------------------
    # HSV変換
    # -------------------------

    hsv = cv2.cvtColor(
        img_bright,
        cv2.COLOR_BGR2HSV
    )

    # -------------------------
    # 赤色の範囲
    # -------------------------

    lower1 = np.array([0, 150, 150])
    upper1 = np.array([8, 255, 255])

    lower2 = np.array([175, 150, 150])
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

    # -------------------------
    # 小さなノイズを除去
    # -------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    return mask


# =========================
# 全画像処理
# =========================

processed = 0
skipped = 0

for fname in sorted(os.listdir(image_dir)):

    # PNG以外は無視
    if not fname.lower().endswith(".png"):
        continue

    # 背景画像は除外
    if fname == "background.png":
        continue

    # Txxx_Pxxx.png のみ
    m = re.match(
        r"T(\d+)_P(\d+)\.png",
        fname
    )

    if not m:
        continue

    tilt = int(m.group(1))
    pan = int(m.group(2))

    # -------------------------
    # Tiltスキャンのみ
    # -------------------------

    if pan != 130:
        continue

    image_path = os.path.join(
        image_dir,
        fname
    )

    print(fname)
    print("読み込み:", image_path)

    # =========================
    # PILで画像を読み込む
    # =========================

    try:

        pil_img = PILImage.open(
            image_path
        ).convert("RGB")

        img = np.array(
            pil_img
        )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

    except Exception as e:

        print(
            f"  → 画像読み込み失敗: {e}"
        )

        skipped += 1
        continue

    # =========================
    # 赤色マスク
    # =========================

    mask = create_red_mask(img)

    # =========================
    # 元画像をコピー
    # =========================

    result = img.copy()

    # =========================
    # 赤色部分を緑色にする
    # =========================

    result[mask > 0] = (
        0,
        255,
        0
    )

    # =========================
    # 保存
    # =========================

    output_path = os.path.join(
        result_dir,
        fname
    )

    try:

        PILImage.fromarray(
            cv2.cvtColor(
                result,
                cv2.COLOR_BGR2RGB
            )
        ).save(output_path)

    except Exception as e:

        print(
            f"  → 保存失敗: {e}"
        )

        skipped += 1
        continue

    # =========================
    # 結果表示
    # =========================

    pixel_count = np.count_nonzero(mask)

    print(
        f"  赤色画素: {pixel_count}"
    )

    print(
        f"  保存: {output_path}"
    )

    processed += 1


# =========================
# 終了
# =========================

print()
print("=========================")
print(f"処理完了: {processed}枚")
print(f"スキップ: {skipped}枚")
print(f"保存先: {result_dir}")
print("=========================")