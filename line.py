import numpy as np
from copy import deepcopy 
from scipy.stats import norm
from random import choice

def line_point_distance(line, point):
    """Calculate minimum euclidean distance from line to point"""
    #I got the formula for finding how close a point is to a line from wikipedia. It requires
    #two points from each line. 
    x0, y0 = point[0], point[1]
    x1, y1 = (line.params["x"], line.params["y"])
    x2, y2 = (line.params["x"] + np.cos(line.params["theta"]), line.params["y"] + np.sin(line.params["theta"]))
    return np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)/np.sqrt((y2-y1)**2 + (x2-x1)**2)

class Line(object):
    """Represent a line and a pointcloud for the line. Provide 
    methods for fitting a line to the pointcloud and quantifying
    the goodness of the fit. 
    
    A line is stored as x, y, and theta under the params object."""

    def __init__(self, params, pointcloud):
        """Save the line parameters, as well as a scale factor to use on the 
        euclidean distance every time it is calculated. """
        self.params = params
        self.pointcloud = pointcloud
    
    def fit_odr(self):
        """Orthogonal distance regress the line"""
        #what is the cost function? total squared error? 
        pass 

    def fit_theil_sen(self):
        """fit a theil sen estimator, modified 
        to use orthogonal distance"""
        pass 

    def fit_p2p(self, cf, n_attempts=10):
        """fit_p2p fits a line by repeatedly selecting a point
        then drawing a line to another point, evaluating this line 
        using the passed cost function"""

        #because we did the kmeans centroid clustering, 
        #a lot of our points are in straight lines. 
        #walls will pass through all those points, 
        #so when searching for the best theta we can 
        #set theta based on other points. 
        tried_thetas = []
        best_found_theta = 0
        best_found_score = 0

        #create line object 
        self.line = Line(self.centerpoint[0], self.centerpoint[1], 0)

        #attempt pointing the line at every point in the self pointcloud
        for point in self.pointcloud:
            #skip iteration if point is the centerpoint
            if np.allclose(point, self.centerpoint):
                continue 
            
            #calculate angle between points
            theta = np.arctan2(point[1] - self.centerpoint[1], point[0] - self.centerpoint[0])
            
            #if the theta is too similar to one that has been tried, skip iteration 
            if tried_thetas and min([np.abs(t-theta) for t in tried_thetas]) < min_theta_resolution:
                continue 
            tried_thetas.append(theta) #record this angle as having been tried
            
            #test this line against the best found line 
            self.line.params["theta"] = theta 
            score = self.line.score(self.pointcloud)
            if score > best_found_score:
                best_found_score = score 
                best_found_theta = theta 
        self.score = best_found_score
        self.line.params["theta"] = best_found_theta
        pass 

    def score_OLS_ODR(self):
        """total of squares of orthodonal distances"""
        pass 

    def score_LAD_ODR(self):
        """total of orthogonal distances"""
        pass 

    def score_chute_length(self, chute_radius=.5, chute_check_resolution=.25, min_chute_region_pop=1):
        """length a sphere can travel along the line until the amount of points found in the sphere falls 
        below min_chute_region_pop, checked every chute_check_resolution. 
        
        returned as a negative value"""
        pass 

    def score_total_norm_cdf(self, distance_scale=.2):
        """ return total of (1 - norm.cdf(distance*self.distance_scale)) for every point
        
        Returned as a negative number. """
        chance = 1 - 1
        pass

    def _calc_point_distances(self):
        """ determine the distance from every point to every line """

    def _theta_subtract(self, theta):
        """Minimum possible radian value of the roation between the passed angles. """
        thetas  = sorted([self.params["theta"], theta])
        while thetas[0] <= thetas[1]:
            prev_theta = thetas[0]
            thetas[0] += np.pi * 2
        difference_a = prev_theta - thetas[1]
        difference_b = thetas[0] - thetas[1]
        return min(difference_a, difference_b)


class LCPopulation(object):
    """Store a collection of lines and a pointcloud.
    
    Each line also stores a pointcloud. The lines pointcloud 
    can be a subset of this larger pointcloud, or the whole thing."""

    def __init__(self, points):
        """points should be a list of points."""
        self.points = points 
        self.pointcloud = Pointcloud(points, local_neighborhood_radius) 
        self.lines = []

    def add_random(self, n, subset_radius):
        """ add n random points using samples from the 
        pointcloud as centerpoints, with the line pointclouds
        the subset of points within subset_radius of this point. """
        #now fill in the population of optimized lines
        for _ in range(n):
            #select a point the line will pass through
            point = choice(points)
            
            #get the points close to this point 
            close_points = self.pointcloud.query_ball_point(point)
            
            self.lines.append(LineCaster(close_points, point))
                 
    def reassign_points(self):
        """Assign every point in the pointcloud to the closest
        line, then replace each line's pointcloud with its assigned 
        points."""
        pass