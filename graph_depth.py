import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

result_dir = r"C:\Users\siro1\k22057nk\研究\卒業研究\result"

os.makedirs(result_dir, exist_ok=True)

# =====================
# 読み込むCSV
# =====================

file = r"C:\Users\siro1\k22057nk\研究\卒業研究\data\T144_P080.csv"

df = pd.read_csv(file)

# =====================
# データ取得
# =====================

x = df["x"]
y = df["y"]

# =====================
# 基準線作成
# =====================

coef = np.polyfit(x, y, 1)

baseline = np.polyval(coef, x)

# =====================
# 深さ計算
# =====================

depth = baseline - y

# =====================
# 溝判定
# =====================

threshold = 10

groove = depth > threshold

# =====================
# グラフ1
# レーザー形状
# =====================

plt.figure(figsize=(10, 5))

plt.plot(
    x,
    y,
    label="Laser Profile"
)

plt.plot(
    x,
    baseline,
    "--",
    label="Baseline"
)

plt.scatter(
    x[groove],
    y[groove]
)

plt.xlabel("x [pixel]")
plt.ylabel("y [pixel]")

plt.title("Laser Profile")

plt.legend()
plt.grid()

plt.savefig(
    os.path.join(result_dir, "laser_profile.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================
# グラフ2
# 深さグラフ
# =====================

plt.figure(figsize=(10, 5))

plt.plot(
    x,
    depth,
    label="Depth"
)

plt.axhline(
    threshold,
    linestyle="--",
    label="Threshold"
)

plt.xlabel("x [pixel]")
plt.ylabel("depth [pixel]")

plt.title("Groove Depth")

plt.legend()
plt.grid()

plt.savefig(
    "groove_depth.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================
# Excel出力
# =====================

result = pd.DataFrame({
    "x": x,
    "y": y,
    "baseline": baseline,
    "depth": depth,
    "groove": groove
})

result.to_excel(
    os.path.join(result_dir, "depth_result.xlsx"),
    index=False
)

# =====================
# 集計
# =====================

summary = pd.DataFrame({
    "max_depth": [depth.max()],
    "mean_depth": [depth.mean()],
    "groove_count": [groove.sum()]
})

summary.to_excel(
    "summary.xlsx",
    index=False
)

# =====================
# 結果表示
# =====================

print("最大深さ =", depth.max())
print("平均深さ =", depth.mean())
print("溝判定点数 =", groove.sum())

print("保存完了")
print("laser_profile.png")
print("groove_depth.png")
print("depth_result.xlsx")
print("summary.xlsx")