from scipy.spatial.distance import euclidean


def calculate_centroid(points, type_of_distance):
    """
    Calculates a centroid of a cluster.

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
