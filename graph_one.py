import pandas as pd
import matplotlib.pyplot as plt

# 読み込むCSV
file = r"data\T130_P080.csv"

df = pd.read_csv(file)

print(df.head())
print(df.shape)

plt.figure(figsize=(10,5))

plt.plot(
    df["x"],
    df["y"]
)

groove = df[df["groove"] == 1]

plt.scatter(
    groove["x"],
    groove["y"]
)

plt.xlabel("x [pixel]")
plt.ylabel("y [pixel]")

plt.title("Laser Profile with Groove Points")

plt.grid()

plt.savefig(
    "profile_T130_P080.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()