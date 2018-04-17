from . import KMeans


class BisectingKMeans:
    def __init__(self, num_clusters: int, type_of_distance: str, seed: int):
        self.num_clusters = num_clusters
        self.type_of_distance = type_of_distance
        self.seed = seed

    def fit(self, data):
        clusters = [data]
        # Remove a cluster from the list of clusters
        cluster = clusters.pop()
        num_trials = 5
        for i in range(num_trials):
            kmeans = KMeans(num_clusters=2, type_of_distance=self.type_of_distance, seed=self.seed)
            kmeans.fit(data)

