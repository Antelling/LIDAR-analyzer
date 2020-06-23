from line import line_point_distance
from scipy.spatial import KDTree

class Pointcloud(object):
    def __init__(self, points, local_neighborhood_radius):
        self.points = points 
        
        self.kdtree = {}
        self.kdtree_sync = False
        self.local_neighborhood_radius = local_neighborhood_radius
        
    def segment_over_line(self, line):
        explained = []
        unexplained = []
        for point in self.points:
            distance = line_point_distance(line, point)
            if distance < self.local_neighborhood_radius: 
                explained.append(point)
            else:
                unexplained.append(point)
        return explained, unexplained
    
    def query_ball_point(self, point):
        #ensure kdtree is up to date 
        if not self.kdtree_sync:
            self.kdtree = KDTree(self.points)
            self.kdtree_sync = True
        
        close_point_indexes = self.kdtree.query_ball_point([point], self.local_neighborhood_radius)[0]
        return [self.points[i] for i in close_point_indexes]