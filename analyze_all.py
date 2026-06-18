import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import re

# =====================
# パス設定
# =====================

data_dir  = r"C:\Users\siro1\k22057nk\研究\卒業研究\data"
result_dir = r"C:\Users\siro1\k22057nk\研究\卒業研究\result"

os.makedirs(result_dir, exist_ok=True)

threshold = 10

# =====================
# CSV読み込み＆集計
# =====================

tilt_records = []  # かかと〜つま先方向
pan_records  = []  # 左右方向

for fname in sorted(os.listdir(data_dir)):
    if not fname.endswith(".csv"):
        continue

    m = re.match(r"T(\d+)_P(\d+)\.csv", fname)
    if not m:
        continue

    tilt_angle = int(m.group(1))
    pan_angle  = int(m.group(2))

    fpath = os.path.join(data_dir, fname)

    try:
        df = pd.read_csv(fpath)
        if df.empty or "x" not in df.columns or "y" not in df.columns:
            continue

        x = df["x"].values
        y = df["y"].values

        if len(x) < 3:
            continue

        # 基準線・深さ計算（graph_depth.pyと同じロジック）
        coef     = np.polyfit(x, y, 1)
        baseline = np.polyval(coef, x)
        depth    = baseline - y
        groove   = depth > threshold

        rec = {
            "file":        fname,
            "tilt":        tilt_angle,
            "pan":         pan_angle,
            "max_depth":   depth.max(),
            "mean_depth":  depth.mean(),
            "groove_ratio": groove.mean(),   # 溝判定点の割合（0〜1）
            "groove_count": groove.sum(),
            "total_points": len(x),
        }

        # Panスキャン：Tilt=070固定
        if tilt_angle == 70:
            pan_records.append(rec)
        # Tiltスキャン：Pan=080固定
        elif pan_angle == 80:
            tilt_records.append(rec)

    except Exception as e:
        print(f"スキップ: {fname} → {e}")

tilt_df = pd.DataFrame(tilt_records).sort_values("tilt").reset_index(drop=True)
pan_df  = pd.DataFrame(pan_records).sort_values("pan").reset_index(drop=True)

print(f"Tiltスキャン: {len(tilt_df)}枚, Panスキャン: {len(pan_df)}枚")

# =====================
# グラフ1
# Tiltスキャン：かかと〜つま先の溝スコア
# =====================

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax = axes[0]
ax.bar(
    tilt_df["tilt"],
    tilt_df["groove_ratio"],
    width=1.5,
    color=[
        "tomato" if v < 0.05 else "steelblue"
        for v in tilt_df["groove_ratio"]
    ]
)
ax.axhline(0.05, linestyle="--", color="red", label="要注意ライン(5%)")
ax.set_xlabel("Tilt角度（小=かかと側 / 大=つま先側）")
ax.set_ylabel("溝スコア（溝判定点の割合）")
ax.set_title("かかと〜つま先方向の溝分布")
ax.legend()
ax.grid(axis="y")

# グラフ2
# Panスキャン：左右の溝スコア
ax = axes[1]
ax.bar(
    pan_df["pan"],
    pan_df["groove_ratio"],
    width=1.5,
    color=[
        "tomato" if v < 0.05 else "steelblue"
        for v in pan_df["groove_ratio"]
    ]
)
ax.axhline(0.05, linestyle="--", color="red", label="要注意ライン(5%)")
ax.set_xlabel("Pan角度（小=左側 / 大=右側）")
ax.set_ylabel("溝スコア（溝判定点の割合）")
ax.set_title("左右方向の溝分布")
ax.legend()
ax.grid(axis="y")

plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "groove_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# =====================
# グラフ3
# Tiltスキャン：最大深さの推移
# =====================

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax = axes[0]
ax.plot(tilt_df["tilt"], tilt_df["max_depth"], marker="o", color="steelblue")
ax.set_xlabel("Tilt角度（小=かかと側 / 大=つま先側）")
ax.set_ylabel("最大深さ [pixel]")
ax.set_title("かかと〜つま先方向の最大溝深さ")
ax.grid()

ax = axes[1]
ax.plot(pan_df["pan"], pan_df["max_depth"], marker="o", color="darkorange")
ax.set_xlabel("Pan角度（小=左側 / 大=右側）")
ax.set_ylabel("最大深さ [pixel]")
ax.set_title("左右方向の最大溝深さ")
ax.grid()

plt.tight_layout()
plt.savefig(
    os.path.join(result_dir, "max_depth_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# =====================
# すり減り判定
# =====================

def wear_judge(groove_ratio_series, label):
    mean_score = groove_ratio_series.mean()
    low_ratio  = (groove_ratio_series < 0.05).mean()  # 要注意ゾーンの割合

    if mean_score >= 0.15:
        level = "◎ 良好"
    elif mean_score >= 0.08:
        level = "○ 普通"
    elif mean_score >= 0.03:
        level = "△ やや摩耗"
    else:
        level = "✕ 要交換"

    print(f"\n【{label}】")
    print(f"  平均溝スコア : {mean_score:.3f}")
    print(f"  要注意ゾーン : {low_ratio*100:.1f}% のスキャンで溝不足")
    print(f"  判定         : {level}")
    return mean_score, level

tilt_score, tilt_level = wear_judge(tilt_df["groove_ratio"], "かかと〜つま先方向")
pan_score,  pan_level  = wear_judge(pan_df["groove_ratio"],  "左右方向")

# かかとゾーン（小さいTilt角）とつま先ゾーン（大きいTilt角）を比較
mid = tilt_df["tilt"].median()
heel_score = tilt_df[tilt_df["tilt"] <= mid]["groove_ratio"].mean()
toe_score  = tilt_df[tilt_df["tilt"] >  mid]["groove_ratio"].mean()

print(f"\n【部位別比較】")
print(f"  かかと側の溝スコア : {heel_score:.3f}")
print(f"  つま先側の溝スコア : {toe_score:.3f}")
if heel_score < toe_score * 0.7:
    print("  → かかと側が特にすり減っています")
elif toe_score < heel_score * 0.7:
    print("  → つま先側が特にすり減っています")
else:
    print("  → かかと・つま先は均等にすり減っています")

# =====================
# Excel出力（集計）
# =====================

with pd.ExcelWriter(os.path.join(result_dir, "all_summary.xlsx")) as writer:
    tilt_df.to_excel(writer, sheet_name="Tiltスキャン", index=False)
    pan_df.to_excel(writer,  sheet_name="Panスキャン",  index=False)

    summary = pd.DataFrame({
        "項目":   ["Tilt平均溝スコア", "Pan平均溝スコア",
                   "かかと側溝スコア", "つま先側溝スコア",
                   "Tilt判定",         "Pan判定"],
        "値":     [f"{tilt_score:.3f}", f"{pan_score:.3f}",
                   f"{heel_score:.3f}", f"{toe_score:.3f}",
                   tilt_level,          pan_level],
    })
    summary.to_excel(writer, sheet_name="判定サマリ", index=False)

print(f"\n保存完了")
print(f"  groove_distribution.png")
print(f"  max_depth_distribution.png")
print(f"  all_summary.xlsx")