import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

fileName = "./results/worstScores.csv"
df = pd.read_csv(fileName, names = ["Image ID", "VS Size"])
print(df)
print("Mean: {}".format(np.mean(df["VS Size"])))
print("Highest: {}".format(np.max(df["VS Size"])))
print("Lowest: {}".format(np.min(df["VS Size"])))
print("Standard Dev: {}".format(np.std(df["VS Size"])))

plt.figure()
plt.hist(df["VS Size"])
plt.xlabel("Version Space Size after Training")
plt.ylabel("Number of Instances")
plt.title("Worst-case VS Ordering")
plt.show()