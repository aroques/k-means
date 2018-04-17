from math import cos
from math import pi
from math import sin
from typing import List
from random import Random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_clusters(clusters: List[List], labels: List[int], centroids: List[List]) -> None:
    """Plot custer data.

    Args:
        clusters: Cluster data to plot.
        labels: The cluster each point belongs to.
        centroids: Centroids of clusters

    Returns:
        None
    """

    # Setup needed to construct plot
    num_clusters = len(set(labels))
    markers = get_markers(num_clusters)
    palette = get_palette(num_clusters)
    columns = ['x', 'y']

    # Get dataframe for data
    df = pd.DataFrame(clusters, columns=columns)
    df['labels'] = pd.Series(labels, index=df.index)  # Add labels as a column for coloring

    # Add centroids to dataframe
    centroids_df = pd.DataFrame(centroids, columns=columns)
    centroids_df['labels'] = ['centroid' for _ in range(len(centroids))]
    df = df.append(centroids_df, ignore_index=True)

    # Plot
    sns.lmplot(*columns, data=df, fit_reg=False, legend=False,
               hue='labels', palette=palette, markers=markers,
               scatter_kws={'s': 50})
    plt.show()


def get_markers(num_clusters):
    random = Random(0)
    markers = ['*', 'o', '^', '+']
    markers = random.sample(markers, num_clusters)
    markers.append('x')
    return markers


def get_palette(num_clusters):
    random = Random(0)
    colors = ['blue', 'orange', 'green', 'purple']
    colors = random.sample(colors, num_clusters)
    colors.append('red')
    return colors


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
