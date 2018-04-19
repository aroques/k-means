from typing import List

from dataviz import generate_clusters
from dataviz import plot_clusters
from kmeans import KMeans, BisectingKMeans
import matplotlib.pyplot as plt


def main():
    num_clusters = 4
    clustered_data = generate_data(num_clusters)
    type_of_distance = 'euclidean'

    #k_means = KMeans(num_clusters=num_clusters, type_of_distance=type_of_distance, seed=0)
    #k_means.fit(clustered_data)

    bk_means = BisectingKMeans(num_clusters=num_clusters, type_of_distance=type_of_distance, seed=0)
    bk_means.fit(clustered_data)

    print(len(clustered_data))
    print(len(bk_means.labels))
    print(len(set(bk_means.labels)))
    print(len(bk_means.centroids))

    plot_clusters(clustered_data, bk_means.centroids, bk_means.labels)
    plt.show()


def generate_data(num_clusters: int) -> List[List]:
    """
    Generates 'clustery' data points
    Args:
        num_clusters: How many clusters to generate

    Returns:
        List of data

    """
    total_points = 20
    pts_per_cluster = int(total_points / num_clusters)
    upper_bound = 100
    lower_bound = 0
    spread = 15
    bounds = (lower_bound+spread, upper_bound-spread)
    return generate_clusters(num_clusters, pts_per_cluster, spread, bounds, bounds)


if __name__ == '__main__':
    main()