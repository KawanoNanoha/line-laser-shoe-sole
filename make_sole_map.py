import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import re

# =====================
# パス設定
# =====================

data_dir   = r"C:\Users\siro1\k22057nk\研究\卒業研究\data"
result_dir = r"C:\Users\siro1\k22057nk\研究\卒業研究\result"

os.makedirs(result_dir, exist_ok=True)

threshold  = 10
img_width  = 640   # カメラ解像度の横幅
img_height = 480   # カメラ解像度の縦幅

# =====================
# Tiltスキャン読み込み
# =====================

tilt_data = {}  # { tilt角: { x: depth } }

for fname in sorted(os.listdir(data_dir)):
    if not fname.endswith(".csv"):
        continue

    m = re.match(r"T(\d+)_P(\d+)\.csv", fname)
    if not m:
        continue

    tilt_angle = int(m.group(1))
    pan_angle  = int(m.group(2))

    # Tiltスキャンのみ（Pan=080固定）
    if pan_angle != 80:
        continue

    fpath = os.path.join(data_dir, fname)

    try:
        df = pd.read_csv(fpath)
        if df.empty or "x" not in df.columns or "y" not in df.columns:
            continue
        if len(df) < 3:
            continue

        x = df["x"].values
        y = df["y"].values

        # baseline基準で深さ再計算
        coef     = np.polyfit(x, y, 1)
        baseline = np.polyval(coef, x)
        depth    = baseline - y

        # x座標 → depth の辞書
        depth_map = {}
        groove_map = {}
        for xi, di in zip(x, depth):
            depth_map[int(xi)]  = di
            groove_map[int(xi)] = 1 if di > threshold else 0

        tilt_data[tilt_angle] = {
            "depth":  depth_map,
            "groove": groove_map,
        }

    except Exception as e:
        print(f"スキップ: {fname} → {e}")

tilt_angles = sorted(tilt_data.keys())
print(f"Tiltスキャン読み込み完了: {len(tilt_angles)}枚")
print(f"Tilt角範囲: {tilt_angles[0]} 〜 {tilt_angles[-1]}")

# =====================
# マップ画像を作成
#
# 横軸 = Tilt角（左=かかと / 右=つま先）
# 縦軸 = x座標（上=左足側 / 下=右足側）
#
# 色:
#   赤   = 溝あり（groove=1）
#   青   = 溝なし（groove=0）
#   黒   = データなし
# =====================

map_w = len(tilt_angles)
map_h = img_width  # x座標の範囲（0〜640）

# depth画像・groove画像の2枚作る
depth_img  = np.zeros((map_h, map_w), dtype=np.float32)
groove_img = np.full((map_h, map_w, 3), 30, dtype=np.uint8)  # 初期値=暗いグレー（データなし）

depth_min =  9999
depth_max = -9999

for col_idx, tilt in enumerate(tilt_angles):
    d = tilt_data[tilt]["depth"]
    g = tilt_data[tilt]["groove"]

    for x_coord, dep in d.items():
        if 0 <= x_coord < map_h:
            depth_img[x_coord, col_idx] = dep
            depth_min = min(depth_min, dep)
            depth_max = max(depth_max, dep)

    for x_coord, grv in g.items():
        if 0 <= x_coord < map_h:
            if grv == 1:
                groove_img[x_coord, col_idx] = [220, 60, 60]    # 赤=溝あり
            else:
                groove_img[x_coord, col_idx] = [60, 100, 180]   # 青=溝なし

# =====================
# 描画
# =====================

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# --- 溝マップ ---
ax = axes[0]
ax.imshow(
    groove_img,
    aspect="auto",
    origin="upper",
    extent=[tilt_angles[0], tilt_angles[-1], img_width, 0]
)
ax.set_xlabel("Tilt angle  (left=heel / right=toe)")
ax.set_ylabel("x coordinate  (top=left side / bottom=right side)")
ax.set_title("Groove Map  (red=groove / blue=flat / dark=no data)")

# かかと・つま先のラベル
ax.text(tilt_angles[0]  + 1, -10, "heel", color="white", fontsize=9)
ax.text(tilt_angles[-1] - 8, -10, "toe",  color="white", fontsize=9)

# --- 深さマップ ---
ax = axes[1]
im = ax.imshow(
    depth_img,
    aspect="auto",
    origin="upper",
    extent=[tilt_angles[0], tilt_angles[-1], img_width, 0],
    cmap="hot",
    vmin=0,
    vmax=max(depth_max, 1)
)
plt.colorbar(im, ax=ax, label="depth [pixel]")
ax.set_xlabel("Tilt angle  (left=heel / right=toe)")
ax.set_ylabel("x coordinate  (top=left side / bottom=right side)")
ax.set_title("Depth Map  (brighter=deeper groove)")

ax.text(tilt_angles[0]  + 1, -10, "heel", color="white", fontsize=9)
ax.text(tilt_angles[-1] - 8, -10, "toe",  color="white", fontsize=9)

plt.tight_layout()

out_path = os.path.join(result_dir, "sole_map.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="black")
plt.show()

print(f"\n保存完了: {out_path}")
print(f"深さ範囲: {depth_min:.1f} 〜 {depth_max:.1f} pixel")

# =====================
# 後からPanスキャンを追加する場合のメモ
#
# pan_data = {}  # { pan角: { y: depth } }
# で同様に読み込み、
# 横軸 = Pan角（左=左足側 / 右=右足側）
# 縦軸 = y座標
# として別レイヤーで重ねる
# =====================