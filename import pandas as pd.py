import pandas as pd
import numpy as np

file = r"C:\Users\siro1\k22057nk\研究\卒業研究\data\T100_P080.csv"
df = pd.read_csv(file)
x = df["x"].values
y = df["y"].values

coef = np.polyfit(x, y, 1)
baseline = np.polyval(coef, x)

print("baseline の範囲:")
print(f"  最小: {baseline.min():.1f}")
print(f"  最大: {baseline.max():.1f}")
print(f"  平均: {baseline.mean():.1f}")
print(f"x の範囲: {x.min()} 〜 {x.max()}")