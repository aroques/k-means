from typing import List
from scipy.spatial.distance import euclidean, cityblock
from math import isclose
from random import Random


class KMeans:
    def __init__(self, num_clusters: int, type_of_distance: str, seed: int):
        self.num_clusters = num_clusters
        self.type_of_distance = type_of_distance
        self.random = Random(seed)
        self.centroids = None
        self.labels_ = None

    def fit(self, data: List[List]) -> None:
        """
        Clusters the data into k clusters. The cluster that each data point belongs
        to is stored in self.labels_.
        Args:
            data: List of data points

        Returns:
            None

        """
        self.centroids = self.__select_initial_centroids(data)

        centroids_moved = True

        while centroids_moved:
            self.labels_ = self.__get_labels(self.centroids, data)

            data_by_cluster = self.__get_data_grouped_by_cluster(data)

            new_centroids = self.__compute_new_centroids(data_by_cluster)

            centroids_moved = self.__centroids_have_moved(new_centroids)

            self.centroids = new_centroids

        if len(set(self.labels_)) != self.num_clusters:
            # We have less than k clusters so raise exception
            raise Exception('We lost a cluster!')

    def __select_initial_centroids(self, data: List) -> List[List]:
        """
        Args:
            data: List of data points

        Returns:
            A random sample of num_cluster data points

        """
        return self.random.sample(data, self.num_clusters)

    def __get_labels(self, centroids: List, data: List) -> List[int]:
        """
        Args:
            centroids: The current centroids in the dataset
            data: List of data points

        Returns:
            List of n labels, where n is the length of data.
            Each integer is the number of the cluster data point belongs to.
            Ex: [0, 1, 0, 3, 2, ..., 1].
            This means data-point 0 belongs to cluster 0, data-point 4 belongs to cluster 2, etc.

        """
        labels = []

        for point in data:
            closest_distance = float('inf')
            closest_centroid_index = -1

            for i, centroid in enumerate(centroids):
                distance = self.__calculate_distance(point, centroid)

                if distance < closest_distance:
                    closest_distance = distance
                    closest_centroid_index = i

            labels.append(closest_centroid_index)

        return labels

    def __calculate_distance(self, p1, p2):
        """
        Calculates the distance between two points.
        Args:
            p1: first point
            p2: second point

        Returns:
            Distance between two points.

        """
        if self.type_of_distance.lower() == 'euclidean':
            return euclidean(p1, p2)
        else:
            return cityblock(p1, p2)

    def __get_data_grouped_by_cluster(self, data: List[List]) -> List[List]:
        """
        Args:
            data: List of data points

        Returns:
            data_by_cluster: List of data-points that have been grouped by cluster.
              Index 0 contains list of data points that belong to cluster 0,
              Index 1 contains list of data points that belong to cluster 1, etc.

        """
        data_by_cluster = [[] for _ in range(self.num_clusters)]

        for i, pt in enumerate(data):
            cluster = self.labels_[i]
            data_by_cluster[cluster].append(pt)

        return data_by_cluster

    def __compute_new_centroids(self, data_by_cluster: List[List]) -> List[List]:
        """
        Computes new centroids by calculating each center of all the points
        that belong to a cluster.
        Args:
            data_by_cluster: List of data that has been grouped by cluster.

        Returns:
            centroids: A list of new centroids.

        """
        centroids = []
        for points in data_by_cluster:
            new_centroid = self.__calculate_new_centroid(points)
            centroids.append(list(new_centroid))
        return centroids

    def __calculate_new_centroid(self, points):
        """
        Calculates a new centroid of a cluster
        Args:
            points: Points that belong to a given cluster

        Returns:
            A new centroid

        """
        if self.type_of_distance == 'euclidean':
            sum_of_points = [sum(x) for x in zip(*points)]
            return map(lambda x: x / len(points), sum_of_points)
        else:
            geometric_medoid =  \
                min(map(lambda p1: (p1, sum(map(lambda p2: euclidean(p1, p2), points))), points), key=lambda x: x[1])[0]
            return geometric_medoid

    def __centroids_have_moved(self, new_centroids: List[List]) -> bool:
        """
        Args:
            new_centroids: Centroids that have been computed the calling loop

        Returns:
            centroids_moved: Whether or not any of the new centroids are significantly different
            than the previous centroids. If the new centroids are significantly different, then
            we say that the 'centroids have moved'.

        """
        centroids_moved = False
        for i, new_centroid in enumerate(new_centroids):
            for j, pt in enumerate(new_centroid):
                equal = isclose(new_centroid[j], self.centroids[i][j], abs_tol=0.00001)
                if not equal:
                    centroids_moved = True
                    break
            if centroids_moved:
                break
        return centroids_moved

    @property
    def total_sum_squared_error(self):
        return None