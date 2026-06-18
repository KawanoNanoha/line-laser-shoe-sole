import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

files = glob.glob("data/*.csv")

angles = []
groove_counts = []

for file in files:

    df = pd.read_csv(file)

    groove_count = df["groove"].sum()

    name = os.path.basename(file)

    angles.append(name)
    groove_counts.append(groove_count)

plt.figure(figsize=(12,6))

plt.bar(range(len(angles)), groove_counts)

plt.xticks(
    range(len(angles)),
    angles,
    rotation=90
)

plt.ylabel("Groove Count")
plt.xlabel("Scan Angle")

plt.title("Groove Detection Result")

plt.tight_layout()

plt.savefig("groove_count.png")

plt.show()