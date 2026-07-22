import cv2
import pandas as pd
import numpy as np
import os
import re
from PIL import Image

#==============================
# フォルダ
#==============================

image_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\image"
depth_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\depth"
result_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\result"

os.makedirs(result_dir, exist_ok=True)

#==============================
# 背景画像
#==============================

background = "T050_P130.png"

img = Image.open(
    os.path.join(image_dir, background)
)

canvas = cv2.cvtColor(
    np.array(img),
    cv2.COLOR_RGB2BGR
)

#==============================
# パラメータ
#==============================

threshold = 8

#==============================
# depth CSV
#==============================

files = sorted(os.listdir(depth_dir))

count = 0

for file in files:

    if not file.endswith(".csv"):
        continue

    print(file)

    path = os.path.join(depth_dir, file)

    df = pd.read_csv(path)

    if len(df) == 0:
        continue

    # depthが大きい点だけ
    groove = df[df["depth"] > threshold]

    if len(groove) == 0:
        continue

    #==========================
    # ファイル名からTilt取得
    #==========================

    m = re.search(r"T(\d+)_P(\d+)", file)

    if m is None:
        continue

    tilt = int(m.group(1))

    #==========================
    # 緑点を描画
    #==========================

    for _, row in groove.iterrows():

        x = int(row["x"])
        y = int(row["y"])

        # ★ここが重要
        #
        # Tiltごとに少しずつ
        # 横へずらして描画する
        #

        draw_x = x + (tilt - 50)

        draw_y = y

        if 0 <= draw_x < canvas.shape[1]:

            cv2.circle(
                canvas,
                (draw_x, draw_y),
                2,
                (0,255,0),
                -1
            )

            count += 1

print()

print("溝点:",count)

#==============================
# 保存
#==============================

outfile = os.path.join(
    result_dir,
    "mapping.png"
)

Image.fromarray(
    cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)
).save(outfile)

print()

print("保存しました")
print(outfile)

cv2.imshow("mapping",canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()