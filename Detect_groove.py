import pandas as pd
import os

# =========================
# パス設定
# =========================

base_dir = r"C:\Users\siro1\k22057nk\研究\line-laser-shoe-sole"

points_dir = os.path.join(base_dir, "points")
groove_dir = os.path.join(base_dir, "groove")

# grooveフォルダを作成
os.makedirs(groove_dir, exist_ok=True)

print("点群フォルダ:", points_dir)
print("点群フォルダ存在:", os.path.exists(points_dir))

print("溝フォルダ:", groove_dir)
print("溝フォルダ存在:", os.path.exists(groove_dir))

# =========================
# 溝判定の閾値
# =========================

threshold = 12

# =========================
# CSV一覧
# =========================

files = sorted(os.listdir(points_dir))

processed = 0

for fname in files:

    if not fname.lower().endswith(".csv"):
        continue

    print("\n処理:", fname)

    path = os.path.join(points_dir, fname)

    try:

        df = pd.read_csv(path)

        if len(df) < 2:
            print("点数不足")
            continue

        # x,yを取得
        x = df["x"].astype(int).values
        y = df["y"].astype(int).values

        groove_points = []

        # =========================
        # 隣接点のx差を見る
        # =========================

        for i in range(1, len(x)):

            dx = abs(x[i] - x[i - 1])

            if dx > threshold:

                groove_points.append([
                    x[i],
                    y[i],
                    dx
                ])

        # =========================
        # CSV保存
        # =========================

        out_df = pd.DataFrame(
            groove_points,
            columns=["x", "y", "dx"]
        )

        out_path = os.path.join(
            groove_dir,
            fname
        )

        out_df.to_csv(
            out_path,
            index=False
        )

        print(
            f"点群: {len(df)}点 → "
            f"溝候補: {len(out_df)}点"
        )

        print("保存:", out_path)

        processed += 1

    except Exception as e:

        print("エラー:", e)

# =========================
# 終了
# =========================

print("\n=========================")
print("処理完了")
print("処理枚数:", processed)
print("=========================")