from typing import Tuple, List
from math import sqrt
import random
import matplotlib.pyplot as plt

import config


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


def distance_for_each_k(rows, k_range):
    distance = euclidean_squared
    distances = []
    for k in k_range:
        _, best_distance = kcluster(rows, distance, k)
        distances.append(best_distance)
    return distances


# ......... K-MEANS ..........
def kcluster(rows, distance, k=config.k_for_clusters) -> Tuple[List, float]:
    ranges = []
    for i in range(len(rows[0])):
        col_values = []
        for row in rows:
            col_values.append(row[i])
        ranges.append((min(col_values), max(col_values)))

    lowest_distance = float('inf')
    best_centroids = None
    best_matches = None

    for _ in range(config.iterations):
        centroids = fill_centroids(k, ranges)
        last_matches = None

        for t in range(100):
            best_matches = group_rows(rows, centroids, distance)

            if best_matches == last_matches:
                break
            last_matches = best_matches

            centroids = update_centroid(rows, best_matches, k)

        total_distance = get_total_distance(rows, best_matches, centroids, distance)

        if total_distance < lowest_distance:
            lowest_distance = total_distance
            best_centroids = centroids
            best_matches = best_matches

    return best_centroids, lowest_distance


def fill_centroids(k, ranges) -> List:
    centroids = []
    for i in range(k):
        centroid = []
        for r in ranges:
            centroid.append(random.uniform(r[0], r[1]))
        centroids.append(centroid)
    return centroids


def group_rows(rows, centroids, distance) -> List:
    """
    This function groups rows into k clusters
    """
    best_matches = []
    for _ in range(len(centroids)):
        best_matches.append([])

    for j in range(len(rows)):
        row = rows[j]
        best_match = 0
        min_dist = distance(centroids[0], row)
        for i in range(1, len(centroids)):
            d = distance(centroids[i], row)
            if d < min_dist:
                best_match = i
                min_dist = d
        best_matches[best_match].append(j)
    return best_matches


def update_centroid(rows, best_matches, k) -> List:
    new_centroids = []
    for i in range(k):
        # Make sure the list of best_matches indexes has elements
        if not best_matches or i >= len(best_matches) or len(best_matches[i]) == 0:
            continue

        averages = [0.0] * len(rows[0])
        for row_id in best_matches[i]:
            for m in range(len(rows[0])):
                averages[m] += rows[row_id][m]

        # Calculate the average for each column
        for j in range(len(averages)):
            averages[j] = averages[j] / len(best_matches[i])
        new_centroids.append(averages)
    return new_centroids


def get_total_distance(rows, best_matches, centroids, distance) -> float:
    total_distance = 0
    for i in range(len(centroids)):
        for row_id in best_matches[i]:
            total_distance += distance(rows[row_id], centroids[i])
    return total_distance


# ...........MAIN.............
def main():
    test_clustering(config.FILE3)  # blogdata.txt
    test_clustering(config.FILE4)  # blogdata_full.txt


def test_clustering(filename) -> None:
    row_names, headers, data = readfile(filename)

    distances = distance_for_each_k(data, config.k_range)

    config.print_line(f"Clustering of {filename} file")
    print("Total distances for different k values:")
    for k, dist in zip(config.k_range, distances):
        print(f"k = {k}: Total distance = {dist}")

    if config.SHOW_PLOTS:
        print_plot(config.k_range, distances)


def print_plot(k_range, distances):
    plt.plot(k_range, distances, '-o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Total distance')
    plt.title('Elbow Method For Optimal k')
    plt.show()


if __name__ == '__main__':
    main()
