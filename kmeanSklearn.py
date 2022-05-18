from cProfile import label
import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

pd.options.mode.chained_assignment = None

kCluster = 5

data = pd.read_csv("./dataset/Mall_Customers.csv")
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

model = KMeans(kCluster)
labels = model.fit(X)
centroids = model.cluster_centers_


annualIncomes = []
spendingScores = []
for c in centroids:
    annualIncomes.append(c[0])
    spendingScores.append(c[1])
centroidDF = pd.DataFrame(
    list(zip(annualIncomes, spendingScores)), columns=["Annual Income (k$)", "Spending Score (1-100)"]
)


# Display result
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plotColors = ["b", "g", "c", "m", "y", "k", "b", "g"]
for i in range(kCluster):
    data = X[labels.labels_ == i]
    plt.plot(
        data["Annual Income (k$)"],
        data["Spending Score (1-100)"],
        plotColors[i] + "^",
        markersize=4,
        label="cluster_" + str(i),
    )
    plt.plot(
        centroidDF["Annual Income (k$)"],
        centroidDF["Spending Score (1-100)"],
        "ro",
        markersize=8,
        label="centroid_" + str(i),
    )
plt.show()

#  Result
print("Final centroids:\n", centroidDF)
X["Cluster"] = labels.labels_.tolist()
print(X.groupby("Cluster").size())
