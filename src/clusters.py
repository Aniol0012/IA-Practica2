from typing import Tuple, List
from math import sqrt
import random


def readfile(filename: str) -> Tuple[List, List, List]:
    headers = None
    row_names = list()
    data = list()

    with open(filename) as file_:
        for line in file_:
            values = line.strip().split("\t")
            if headers is None:
                headers = values[1:]
            else:
                row_names.append(values[0])
                data.append([float(x) for x in values[1:]])
    return row_names, headers, data


# .........DISTANCES........
# They are normalized between 0 and 1, where 1 means two vectors are identical
def euclidean(v1, v2):
    distance = 0
    for i in range(len(v1)):
        distance += (v1[i] - v2[i]) ** 2
    return 1 / (1 + distance)


def euclidean_squared(v1, v2):
    return euclidean(v1, v2) ** 2


def pearson(v1, v2):
    # Simple sums
    sum1 = sum(v1)
    sum2 = sum(v2)
    # Sums of squares
    sum1sq = sum([v ** 2 for v in v1])
    sum2sq = sum([v ** 2 for v in v2])
    # Sum of the products
    products = sum([a * b for (a, b) in zip(v1, v2)])
    # Calculate r (Pearson score)
    num = products - (sum1 * sum2 / len(v1))
    den = sqrt((sum1sq - sum1 ** 2 / len(v1)) * (sum2sq - sum2 ** 2 / len(v1)))
    if den == 0:
        return 0
    return 1 - num / den


# ........HIERARCHICAL........
class BiCluster:
    def __init__(self, vec, left=None, right=None, dist=0.0, id=None):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = dist


def hcluster(rows, distance=pearson):
    distances = {}  # Cache of distance calculations
    currentclustid = -1  # Non original clusters have negative id

    # Clusters are initially just the rows
    clust = [BiCluster(row, id=i) for (i, row) in enumerate(rows)]

    """
    while ...:  # Termination criterion
        lowestpair = (0, 1)
        closest = distance(clust[0].vec, clust[1].vec)

        # loop through every pair looking for the smallest distance
        for i in range(len(clust)):
            for j in range(i+1, len(clust)):
                distances[(clust[i].id, clust[j].id)] = ...

            # update closest and lowestpair if needed
            ...
        # Calculate the average vector of the two clusters
        mergevec = ...

        # Create the new cluster
        new_cluster = BiCluster(...)

        # Update the clusters
        currentclustid -= 1
        del clust[lowestpair[1]]
        del clust[lowestpair[0]]
        clust.append(new_cluster)
    """

    return clust[0]


def printclust(clust: BiCluster, labels=None, n=0):
    # indent to make a hierarchy layout
    indent = " " * n
    if clust.id < 0:
        # Negative means it is a branch
        print(f"{indent}-")
    else:
        # Positive id means that it is a point in the dataset
        if labels == None:
            print(f"{indent}{clust.id}")
        else:
            print(f"{indent}{labels[clust.id]}")
    # Print the right and left branches
    if clust.left != None:
        printclust(clust.left, labels=labels, n=n + 1)
    if clust.right != None:
        printclust(clust.right, labels=labels, n=n + 1)


# ......... K-MEANS ..........
def kcluster(rows, distance=euclidean_squared, k=4, executions=10):
    ranges = []
    for i in range(len(rows[0])):
        col_values = []
        for row in rows:
            col_values.append(row[i])

        min_value = min(col_values)
        max_value = max(col_values)
        ranges.append((min_value, max_value))

    lowest_distance = float('inf')
    best_centroids = None
    best_matches = None

    for _ in range(executions):
        # Start with k randomly placed centroids
        centroids = []
        for j in range(k):
            centroid = []
            for i in range(len(rows[0])):
                centroid.append(random.uniform(ranges[i][0], ranges[i][1]))
            centroids.append(centroid)

        last_matches = None
        for t in range(100):
            best_matches = [[] for _ in range(k)]

            # Find which centroid is the closest for each row
            for j in range(len(rows)):
                row = rows[j]
                best_match = 0
                for i in range(k):
                    d = distance(centroids[i], row)
                    if d < distance(centroids[best_match], row):
                        best_match = i
                best_matches[best_match].append(j)

            if best_matches == last_matches:
                break
            last_matches = best_matches

            # Move the centroids to the average of their members
            for i in range(k):
                if len(best_matches[i]) > 0:
                    avgs = [0.0] * len(rows[0])
                    for row_id in best_matches[i]:
                        for m in range(len(rows[0])):
                            avgs[m] += rows[row_id][m]
                    for j in range(len(avgs)):
                        avgs[j] /= len(best_matches[i])
                    centroids[i] = avgs

        # Calculate the total distance from all the points to their respective centroids
        total_distance = 0
        for i in range(k):
            for row_id in best_matches[i]:
                total_distance += distance(rows[row_id], centroids[i])

        if total_distance < lowest_distance:
            lowest_distance = total_distance
            best_centroids = centroids
            best_matches = best_matches

    return best_centroids, best_matches
