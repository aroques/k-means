from scipy.spatial.distance import euclidean, cityblock
from typing import List
from itertools import chain


def calculate_centroid(points, type_of_distance):
    """Calculates a centroid of a cluster.

    Args:
        points: Points that belong to a given cluster
        type_of_distance: The distance measure to use: Euclidean or Manhattan.

    Returns:
        A new centroid
    """
    if type_of_distance == 'euclidean':
        sum_of_points = [sum(x) for x in zip(*points)]
        return list(map(lambda x: x / len(points), sum_of_points))
    else:
        geometric_medoid = \
            min(map(lambda p1: (p1, sum(map(lambda p2: euclidean(p1, p2), points))), points), key=lambda x: x[1])[0]
        return list(geometric_medoid)


def get_intercluster_distances(data_by_cluster: List[List], type_of_distance) -> List[float]:
    """Get the distances between points in different clusters.

    Args:
        data_by_cluster: data grouped by cluster.
        labels: The cluster each point belongs to.
        type_of_distance: Type of distance measure to user: euclidean or manhattan distance.

    Returns:
        A list of inter-cluster distances.
    """
    inter_cluster_distances = []
    for cluster in data_by_cluster:
        for point in cluster:
            other_clusters = data_by_cluster[:]  # Copy cluster data before removing cluster
            other_clusters.remove(cluster)
            points_in_other_clusters = chain.from_iterable(other_clusters)  # Flatten other clusters
            for point_in_other_cluster in points_in_other_clusters:
                inter_cluster_distance = calculate_distance(point, point_in_other_cluster, type_of_distance)
                inter_cluster_distances.append(inter_cluster_distance)
    return inter_cluster_distances


def calculate_distance(p1: List, p2: List, type_of_distance) -> float:
    """Calculates the distance between two points.

    Args:
        p1: first point
        p2: second point
        type_of_distance: type of distance measure to use

    Returns:
        Distance between two points.
    """
    if type_of_distance.lower() == 'euclidean':
        return euclidean(p1, p2)
    else:
        return cityblock(p1, p2)


def total_error(cluster_error):
    """Computes the sum of each intra-cluster error.

    Args:
        cluster_error: Each cluster's intra-cluster error

    Returns:
        The sum of each intra-cluster error
    """
    return sum(cluster_error)


def print_report(kmeans):
    print('Clustering Report - Bisecting K-means')
    intercluster_distances = get_intercluster_distances(kmeans.data_by_cluster, kmeans.type_of_distance)
    max_intercluster = max(intercluster_distances)
    min_intercluster = min(intercluster_distances)
    print('k: {}, type of distance: {}'.format(kmeans.num_clusters, kmeans.type_of_distance))
    print('sum of intra-cluster distance error: {}'.format(total_error(kmeans.cluster_error)))
    print('max inter-cluster distance: {}'.format(max_intercluster))
    print('min inter-cluster distance: {}'.format(min_intercluster))

