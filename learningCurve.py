import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

techniqueCodes = ["rn", "lc", "en", "sm", "rc", "ve", "kl"]
techniqueNames = [
    "Random", "Least Confidence", "Entropy", "Smallest Margin",
    "Ratio Confidence", "Vote Entropy", "KL Divergence"
]
techniqueColors = [
    "tab:blue", "tab:orange", "tab:green",
    "tab:red", "tab:purple", "tab:brown",
    "tab:pink", "tab:gray", "tab:olive",
    "tab:cyan"
]

techniqueDict = dict()
for i in range(len(techniqueCodes)):
    currDict = dict()
    currDict["code"] = techniqueCodes[i]
    currDict["title"] = techniqueNames[i]
    currDict["color"] = techniqueColors[i]
    currDict["train"], currDict["total"], currDict["test"] = [0.844], [0.732], [0.658]
    currDict["fracs"] = [0]
    currDict["fileName"] = "./results/agg_{}.csv".format(techniqueCodes[i])
    techniqueDict[techniqueCodes[i]] = currDict

techniquesRequired = ["rn", "lc", "en", "sm", "rc", "ve", "kl"]
# techniquesRequired = []
# while True:
#     print("Enter the technique: rn/lc/en/sm/rc/ve/kl/quit")
#     entered = str(input())
#     if entered == "quit":
#         break
#     techniquesRequired.append(entered)

for technique in techniquesRequired:
    currDict = techniqueDict[technique]
    df = pd.read_csv(currDict["fileName"], sep = " ", header = None)
    currDict["fracs"] = currDict["fracs"] + list(df.iloc[:, 0])
    currDict["train"] = currDict["train"] + list(df.iloc[:, 1])
    currDict["total"] = currDict["total"] + list(df.iloc[:, 2])
    currDict["test"] = currDict["test"] + list(df.iloc[:, 3])

print(
    np.max(np.array(techniqueDict["ve"]["total"]) - np.array(techniqueDict["rn"]["total"])),
    np.max(np.array(techniqueDict["ve"]["test"]) - np.array(techniqueDict["rn"]["test"])),
    np.max(np.array(techniqueDict["sm"]["total"]) - np.array(techniqueDict["rn"]["total"])),
    np.max(np.array(techniqueDict["ve"]["test"]) - np.array(techniqueDict["rn"]["test"]))
)

# plt.figure()
# for technique in techniquesRequired:
#     currDict = techniqueDict[technique]
#     plt.plot(
#         currDict["fracs"], currDict["total"],
#         label = currDict["title"], color = currDict["color"]
#     )
# plt.xlabel("Queried Instances - Fraction of Total Pool")
# plt.ylabel("Accuracy")
# plt.title("Training Pool Accuracy against Queried Instances")
# plt.legend()
# plt.show()

# plt.figure()
# for technique in techniquesRequired:
#     currDict = techniqueDict[technique]
#     plt.plot(
#         currDict["fracs"], currDict["test"],
#         label = currDict["title"], color = currDict["color"]
#     )
# plt.xlabel("Queried Instances - Fraction of Total Pool")
# plt.ylabel("Accuracy")
# plt.title("Test Accuracy against Queried Instances")
# plt.legend()
# plt.show()

# plt.figure()
# for technique in techniquesRequired:
#     currDict = techniqueDict[technique]
#     plt.plot(
#         currDict["fracs"], currDict["train"],
#         label = currDict["title"], color = currDict["color"]
#     )
# plt.xlabel("Queried Instances - Fraction of Total Pool")
# plt.ylabel("Accuracy")
# plt.title("Train Accuracy against Queried Instances")
# plt.legend()
# plt.show()