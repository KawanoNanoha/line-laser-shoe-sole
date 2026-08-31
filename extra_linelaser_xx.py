import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image


#========================
# フォルダ
#========================

image_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\image"
csv_dir   = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\data"

os.makedirs(csv_dir, exist_ok=True)

print(image_dir)
print(os.path.exists(image_dir))
files = sorted(os.listdir(image_dir))

for file in files:

    if not file.endswith(".png"):
        continue

    print(file)

    path = os.path.join(image_dir, file)

    print(path)

    try:
        img = np.array(Image.open(path))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.convertScaleAbs(img, alpha=5.0, beta=30)
    except Exception as e:
        print(e)
        continue

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2HSV
    )

    #========================
    # 赤色抽出
    #========================

    lower1 = np.array([0,80,80])
    upper1 = np.array([15,255,255])

    lower2 = np.array([160,80,80])
    upper2 = np.array([180,255,255])

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

    kernel = np.ones((3,3),np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    #========================
    # 赤さ画像
    #========================

    red = img[:,:,2]

    h,w = mask.shape

    result=[]

    #========================
    # 各列を調べる
    #========================

    for x in range(w):

        ys = np.where(mask[:,x]>0)[0]

        if len(ys)==0:
            continue

        values = red[ys,x]

        index = np.argmax(values)

        y = ys[index]

        result.append([x,y])

    df = pd.DataFrame(
        result,
        columns=["x","y"]
    )

    out = os.path.join(
        csv_dir,
        file.replace(".png",".csv")
    )

    df.to_csv(
        out,
        index=False
    )

    print("保存:",out)

print("終了")