from . import KMeans
from typing import List
from .utils import calculate_centroid, get_intercluster_distances


class BisectingKMeans:
    def __init__(self, num_clusters: int, type_of_distance: str, seed: int):
        self.num_clusters = num_clusters
        self.type_of_distance = type_of_distance
        self.seed = seed
        self.centroids = None
        self.data_by_cluster = None
        self.cluster_error = [0 for _ in range(self.num_clusters)]

    def fit(self, data: List[List]) -> None:
        # Initialize list of clusters as
        # 1 cluster that contains all the data points
        clusters_with_error = [(data, 0)]

        # While kmeans_list is smaller then num_clusters
        while len(clusters_with_error) < self.num_clusters:

            clusters_with_error = self.__sort_clusters_with_error(clusters_with_error)

            highest_error_cluster = clusters_with_error.pop()

            kmeans_list = self.__run_bisecting_kmeans(highest_error_cluster[0])

            lowest_error_kmeans = self.__select_lowest_error_kmeans(kmeans_list)

            for i, cluster in enumerate(lowest_error_kmeans.data_by_cluster):
                clusters_with_error.append((cluster, lowest_error_kmeans.cluster_error[i]))

        self.data_by_cluster = self.__get_clusters(clusters_with_error)
        self.__compute_centroids(self.data_by_cluster)
        self.__compute_labels(data, self.data_by_cluster)

        # Compute cluster error
        for i, cluster_with_error in enumerate(clusters_with_error):
            error = cluster_with_error[1]
            self.cluster_error[i] = error


    @staticmethod
    def __select_lowest_error_kmeans(kmeans_list):
        """
        Selects the K Means object that has the lowest error.

        Args:
            kmeans_list: List of K Means objects

        Returns:
            K Means object

        """
        k_means_list = sorted(kmeans_list,
                              key=lambda kmeans: kmeans.total_error,
                              reverse=True)
        return k_means_list.pop()

    @staticmethod
    def __sort_clusters_with_error(clusters_with_error):
        """
        Sorts list of cluster with error tuples.

        Args:
            clusters_with_error: List of (cluster, error) tuples.

        Returns:
            Sorted list of (cluster, error) tuples

        """
        return sorted(clusters_with_error, key=lambda cluster_with_error: cluster_with_error[1])

    def __run_bisecting_kmeans(self, data):
        """
        Runs K Means with k=2 n times.

        Args:
            data: List of data points

        Returns:
            List of K Means objects.

        """
        num_trials = 10
        kmeans_list = []
        for i in range(num_trials):
            kmeans = KMeans(num_clusters=2, type_of_distance=self.type_of_distance, seed=self.seed)
            kmeans.fit(data)
            kmeans_list.append(kmeans)
        return kmeans_list

    def __compute_centroids(self, clusters):
        """Computes centroids

        Args:
            clusters: A list of clusters. A cluster is a list of data points.

        Returns:
            None
        """
        self.centroids = []
        for cluster in clusters:
            centroid = calculate_centroid(cluster, self.type_of_distance)
            self.centroids.append(centroid)

    @staticmethod
    def __get_clusters(clusters_with_error):
        """Returns list of clusters.

        Args:
            clusters_with_error: A list of (cluster, error) tuples.

        Returns:
            A list of clusters. A cluster is a list of data points.
        """
        clusters = []
        for cluster_with_error in clusters_with_error:
            clusters.append(cluster_with_error[0])
        return clusters

    def __compute_labels(self, data, clusters):
        """Computes a list of labels.

        Args:
            data: A list of data points.
            clusters: A list of clusters.

        Returns:
            None
        """
        self.labels = []
        for point in data:
            cluster_index = -1
            for i, cluster in enumerate(clusters):
                if point not in cluster:
                    continue
                cluster_index = i
            self.labels.append(cluster_index)


