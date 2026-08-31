import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#========================
# フォルダ
#========================

data_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\data"

result_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole\depth"

os.makedirs(result_dir, exist_ok=True)

#========================
# CSVを全部処理
#========================

files = sorted(os.listdir(data_dir))

for file in files:

    if not file.endswith(".csv"):
        continue

    print(file)

    path = os.path.join(data_dir,file)

    df = pd.read_csv(path)

    if len(df) < 10:
        continue

    x = df["x"].values
    y = df["y"].values

    #========================
    # 基準線（1次近似）
    #========================

    coef = np.polyfit(y, x, 1)

    baseline = np.polyval(coef, y)

    #========================
    # 深さ
    #========================

    depth = baseline - x

    df["baseline"] = baseline
    df["depth"] = depth

    #========================
    # 溝判定
    #========================

    threshold = 5

    df["groove"] = (depth > threshold).astype(int)

    #========================
    # 保存
    #========================

    out_csv = os.path.join(result_dir,file)

    df.to_csv(out_csv,index=False)

    #========================
    # グラフ
    #========================

    plt.figure(figsize=(10,4))

    plt.plot(x,y,label="Laser")

    plt.plot(
        baseline,
        y,
        "--",
        label="Baseline"
    )

    groove = df[df["groove"]==1]

    plt.scatter(
        groove["x"],
        groove["y"],
        s=10,
        label="Groove"
    )

    plt.gca().invert_yaxis()

    plt.legend()

    plt.grid()

    plt.tight_layout()

    png = os.path.join(
        result_dir,
        file.replace(".csv",".png")
    )

    plt.savefig(
        png,
        dpi=300
    )

    plt.close()

print("全部終了")