from typing import List

from dataviz import generate_clusters
from dataviz import plot_clusters
from kmeans import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    num_clusters = 4
    clustered_data = generate_data(num_clusters)
    k_means = KMeans(num_clusters=num_clusters, type_of_distance='manhattan')
    k_means.fit(clustered_data)
    plot_clusters(clustered_data, k_means.labels_)


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


def plot(data: List[List]) -> None:
    """
    Plot data

    Args:
        data: Data to plot.

    Returns:
        None
    """
    columns = ['x', 'y']
    data = pd.DataFrame(data, columns=columns)
    lm = sns.lmplot(*columns, data=data, fit_reg=False, legend=False)
    lm.fig.suptitle('Unlabeled Data')
    plt.show()


if __name__ == '__main__':
    main()