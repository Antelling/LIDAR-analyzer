from line import line_point_distance
from scipy.spatial import KDTree
from sklearn.cluster import MiniBatchKMeans, KMeans
from random import random
import numpy as np 

class Pointcloud(object):
    def __init__(self, points):
        self.points = points 
        
        #fitting the kdtree is expensive, and only needed 
        #for the query ball tree method. 
        self.kdtree = {}
        self.kdtree_sync = False

    def _ensure_kdtree_synced(self):
        """ensure kdtree is up to date """
        if not self.kdtree_sync:
            self.kdtree = KDTree(self.points)
            self.kdtree_sync = True
    
    def query_ball_point(self, point, radius):
        self._ensure_kdtree_synced()
        
        close_point_indexes = self.kdtree.query_ball_point([point], radius)[0]
        return [self.points[i] for i in close_point_indexes]

    def remove_floor(self, floor=.03):
        self.points = [p for p in self.points if p[2] > floor]
        self.kdtree_sync = False

    def take_percentage(self, percent=.1):
        """randomly prune points to get down to the passed percent"""
        self.points = [p for p in self.points if random() < percent]
        self.kdtree_sync = False

    def take_xy(self):
        """lidar scans are [x y z intensity ring_scan_number]
        We normally only need x and y"""
        self.points = [[p[0], p[1]] for p in self.points]
        self.kdtree_sync = False

    def take_centroids(self, n_means, exact=False):
        """ https://github.com/Dibillilia/AveragedClusterAugmenter """
        if not exact:
            clusterer = MiniBatchKMeans(n_clusters=n_means)
        else:
            clusterer = KMeans(n_clusters=n_means)
        clusterer = clusterer.fit(self.points)
        self.points = clusterer.cluster_centers_ #points is now a numpy array
        self.kdtree_sync = False

    def get_nearest_neighbors(self, k):
        """return an array of arrays containing indexes of 
        the nearest neighbors of the self.points array"""
        self._ensure_kdtree_synced()

        return self.kdtree.query(self.points, k)

    def isolation_forest_filter(self):
        pass 

    def biased_undersample(self, percentile=.6, radius=.1):
        """Make all points have as many neighbors as the percentile's 
        amount of neighbors"""
        self._ensure_kdtree_synced()

        n_neighbors = [len(neighbors) for neighbors in self.kdtree.query_ball_point(self.points, radius)]
        nn_index = list(enumerate(n_neighbors))
        nn_index.sort(key=lambda x: x[1])
        percentile_neighbors = nn_index[int(percentile * len(nn_index))][1]
        selected_indexes = [nn_i[0] for nn_i in nn_index if nn_i[1] < percentile_neighbors or np.random.rand() < percentile_neighbors/nn_i[1]]
        self.points = [self.points[i] for i in selected_indexes]
        self.kdtree_sync = False
        self.points