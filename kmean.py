import pandas as pd
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from pathlib import Path 

pd.options.mode.chained_assignment = None


# Global declaration
outputPath =  Path("./kmean.out.csv")
kCluster = 5

# Extract data from csv file
data = pd.read_csv("./dataset/Mall_Customers.csv")
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Plot data to 2D graph using scatter
plt.scatter(
    X["Annual Income (k$)"],
    X["Spending Score (1-100)"],
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()


def initCentroids():
    return X.sample(n=kCluster)


def visualize(centroids, labels, title):
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title(title)
    plotColors = ["b", "g", "c", "m", "y", "k", "b", "g"]

    for i in range(kCluster):
        data = X[labels == i]
        plt.plot(
            data["Annual Income (k$)"],
            data["Spending Score (1-100)"],
            plotColors[i] + "^",
            markersize=4,
            label="cluster_" + str(i),
        )
        plt.plot(
            centroids["Annual Income (k$)"],
            centroids["Spending Score (1-100)"],
            "ro",
            markersize=8,
            label="centroid_" + str(i),
        )
    # plt.show()


def predictLabels(centroids):
    # Calc distance between each data and currently centroids
    attribute = 1
    for centerIndex, centroidRow in centroids.iterrows():
        distances = []
        for xIndex, xRow in X.iterrows():
            distance = np.sqrt(
                (centroidRow["Annual Income (k$)"] - xRow["Annual Income (k$)"]) ** 2
                + (centroidRow["Spending Score (1-100)"] - xRow["Spending Score (1-100)"]) ** 2
            )
            distances.append(distance)
        X[attribute] = distances  # Extend new attribute 'i'
        attribute = attribute + 1

    # Get new predict labels
    labels = []
    for xIndex, row in X.iterrows():
        minDistance = row[1]
        label = 0
        for i in range(kCluster):
            if row[i + 1] < minDistance:
                minDistance = row[i + 1]
                label = i
        labels.append(label)  # labels run from 0 to K - 1
    X["Cluster"] = labels

    return np.array(labels)


def updateCentroids(labels):
    annualIncomeMeans = []
    spendingScoreMeans = []
    for k in range(kCluster):
        data = X[labels == k]
        mean = np.mean(data, axis=0)
        annualIncomeMeans.append(mean["Annual Income (k$)"])
        spendingScoreMeans.append(mean["Spending Score (1-100)"])

    return pd.DataFrame(
        list(zip(annualIncomeMeans, spendingScoreMeans)), columns=["Annual Income (k$)", "Spending Score (1-100)"]
    )


def isSameCentroids(centroids, newCentroids):
    indexes = []
    for i in range(kCluster):
        indexes.append(i)
    centroids.index = indexes

    diff = (newCentroids["Annual Income (k$)"] - centroids["Annual Income (k$)"]).sum() + (
        newCentroids["Spending Score (1-100)"] - centroids["Spending Score (1-100)"]
    ).sum()

    return diff == 0


def kmeans(init_centes, init_labels):
    centroids = init_centes
    labels = init_labels
    attempt = 0
    while True:
        labels = predictLabels(centroids)
        visualize(centroids, labels, "Attempt " + str(attempt + 1) + ":")
        newCentroids = updateCentroids(labels)
        print(newCentroids)
        if isSameCentroids(centroids, newCentroids):
            break
        centroids = newCentroids
        attempt += 1
    return (centroids, labels)


centroids = initCentroids()
print("Init centroids:\n", centroids)
init_labels = np.zeros(X.shape[0])
centroids, labels = kmeans(centroids, init_labels)
print("Final centroids:\n", centroids)
print("Data with Cluster:\n")
X.groupby("Cluster").apply(print)

# 0           88.200000               17.114286
# 1           86.538462               82.128205
# 2           25.727273               79.363636
# 3           26.304348               20.913043
# 4           55.296296               49.518519
