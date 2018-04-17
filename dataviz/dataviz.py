from math import cos
from math import pi
from math import sin
from typing import List
from random import Random, sample

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_clusters(clusters: List[List], labels: List[int]) -> None:
    """Plot custer data.

    Args:
        clusters: Cluster data to plot.
        labels: The cluster each point belongs to.

    Returns:
        None
    """
    markers = ['*', 'o', '^', '+']
    markers = sample(markers, len(set(labels)) - 1)
    markers.append('x')
    columns = ['x', 'y']
    data = pd.DataFrame(clusters, columns=columns)
    data['labels'] = pd.Series(labels, index=data.index)  # Add labels as a column for coloring
    sns.lmplot(*columns, data=data, fit_reg=False, legend=False,
               hue='labels', palette=['blue', 'orange', 'green', 'purple', 'red'], markers=markers,
               scatter_kws={'s': 50})
    plt.show()


def generate_clusters(num_clusters, pts_per_cluster, spread, bound_for_x, bound_for_y) -> List[List]:
    """Generate random data for clustering.

    Source:
    https://stackoverflow.com/questions/44356063/how-to-generate-a-set-of-random-points-within-a-given-x-y-coordinates-in-an-x

    Args:
        num_clusters: The number of clusters to generate.
        pts_per_cluster: The number of points per cluster to generate.
        spread: The spread of each cluster. Decrease for tighter clusters.
        bound_for_x: The bounds for possible values of X.
        bound_for_y: The bounds for possible values of Y.

    Returns:
        K clusters consisting of N points.
    """
    seed = 0
    r = Random(seed)
    x_min, x_max = bound_for_x
    y_min, y_max = bound_for_y
    clusters = []
    for _ in range(num_clusters):
        x = x_min + (x_max - x_min) * r.random()
        y = y_min + (y_max - y_min) * r.random()
        clusters.extend(generate_cluster(pts_per_cluster, (x, y), spread, seed))
    return clusters


def generate_cluster(num_points, center, spread, seed) -> List[List]:
    """Generates a cluster of random points.

    Source:
    https://stackoverflow.com/questions/44356063/how-to-generate-a-set-of-random-points-within-a-given-x-y-coordinates-in-an-x

    Args:
        num_points: The number of points for the cluster.
        center: The center of the cluster.
        spread: How tightly to cluster the data.
        seed: Seed for random

    Returns:
        A random cluster of consisting of N points.
    """
    x, y = center
    seed = seed ^ int(x * y)  # Modify seed based on x and y so that each cluster is different
    r = Random(seed)
    points = []
    for i in range(num_points):
        theta = 2 * pi * r.random()
        s = spread * r.random()
        point = [x + s * cos(theta), y + s * sin(theta)]
        points.append(point)
    return points
