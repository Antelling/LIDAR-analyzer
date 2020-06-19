"""Theres a scikit-learn-contrib library called imbalanced-learn thats 
meant to be used when you have many more samples for one label than another.
It provides numerous methods for turning lots of points into a smaller set of 
representative points, which we are interested in because raw lidar data gives 
way too many points. 

However, the api expects labeled data, and tries to automatically adjust the 
labels to a passed ratio. We have only one label, so there is no way to control 
how many samples we take. The cluster_centroids method is simple enough (given a 
clusterer) that we can implement it ourselves. """
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from scipy import spatial

def cluster_centroids(x, n_clusters):
    """fit kmeans clusters then return the average of the clusters"""
    #this doesn't work well at all 
    clusterer = MiniBatchKMeans(n_clusters=n_clusters)
    clusterer = clusterer.fit(x)
    groups = {}
    for i, label in enumerate(clusterer.labels_):
        if not label in groups:
            groups[label] = []
        groups[label].append(x[i])
    
    for group in groups:
        groups[group] = np.mean(groups[group], axis=0).tolist()

    return np.array(list(groups.values()))

def min_density_selection(points, n_neighbors, distance):
    """construct a new pointcloud only containing points with at 
    least n_neighbors within distance.
    
    returns a numpy array of selected points"""
    tree = spatial.KDTree(points)
    #kdtree does not copy the points, it just makes an index over them 
    selected_points = []
    for point in points:
        neighbors = tree.query_ball_point([point], distance)[0]
        if len(neighbors) >= n_neighbors:
            selected_points.append(point)

    return np.array(selected_points)