import numpy as np
from copy import deepcopy 
from scipy.stats import norm
from random import choice
from pointcloud import Pointcloud

def line_point_distance(line, point):
    """Calculate minimum euclidrand_choiceean distance from line to point"""
    #I got the formula for finding how close a point is to a line from wikipedia. It requires
    #two points from each line. 
    x0, y0 = point[0], point[1]
    x1, y1 = (line.params["x"], line.params["y"])
    x2, y2 = (line.params["x"] + np.cos(line.params["theta"]), line.params["y"] + np.sin(line.params["theta"]))
    return np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)/np.sqrt((y2-y1)**2 + (x2-x1)**2)

class Line(object):
    """Represent a line and provide methods of doing math involving other lines. 
    
    A line is stored as x, y, and theta under the params object."""

    def __init__(self, x, y, theta, distance_scale=.2):
        """Save the line parameters, as well as a scale factor to use on the 
        euclidean distance every time it is calculated. """
        self.params = {"x": x, "y": y, "theta": theta}
        self.distance_scale = distance_scale
 
    def score(self, points, method=1, chute_resolution=.1, chute_radius=.05, pointcloud=None):
        """The score method accepts a list of points, and returns a value 
        quantifying how good this line fits the passed points. There are 
        different methods for calculating the score. 
        Method is one of:
        
        0: we want score to reflect the summation of the likelihood that a wall occupying this space would 
        trigger a recorded point for each point in the pointcloud.
        This method is intended to reflect the amount of points that are explained by a wall occupying this 
        position. 
        
        1: check how long the starting point can slide along the line before a point is not within 
        localdistance, checked every chute_resolution increment. Returns the amount of increments 
        traveled in both directions.  """
        total = 0

        if method == 0:
            for point in points:
                chance = self._likelihood_of_being_cause_of_point(point)
                total += chance 
            self.most_recent_score = total

        if method == 1:
            for sign in [1, -1]:
                steps = 0
                while True:
                    steps += 1
                    x = np.cos(self.params["theta"] * steps * sign) + self.params["x"]
                    y = np.sin(self.params["theta"] * steps * sign) + self.params["y"]
                    n_neighbors = len(pointcloud.query_ball_point([[x, y]], chute_radius)[0])
                    if n_neighbors:
                        total += 1
                    else:
                        break 

        return total 

    def _likelihood_of_being_cause_of_point(self, point):
        """This function returns how likely it is the passed 
        point would be caused by this line"""
        #LIDAR scans in cylindrical dimensions, with a standard deviation along each 
        #dimension. So, if this method is passed a single LIDAR point, it should 
        #do all the math in cylindrical dimensions scaled according to the standard 
        # deviation. However, we do a lot of pointcloud cleaning before this function 
        #is called, so the points passed will hopefully represent a probability zone 
        #with the same standard deviation along all dimensions. So we can just find the
        #distance to the line then plug the distance into a gaussian cdf function
        
        #get euclidean distance to point
        distance = line_point_distance(self, point)
        
        #convert distance to the likelihood this wall would cause this point
        #FIXME: distance should be scaled based on the standard deviation of the points from the
        #true wall location. I randomly picked the value 2. 
        chance = 1 - norm.cdf(distance*self.distance_scale)
        return chance   

    def _theta_subtract(self, theta):
        """Minimum possible radian value of the roation between the passed angles. """
        thetas  = sorted([self.params["theta"], theta])
        while thetas[0] <= thetas[1]:
            prev_theta = thetas[0]
            thetas[0] += np.pi * 2
        difference_a = prev_theta - thetas[1]
        difference_b = thetas[0] - thetas[1]
        return min(difference_a, difference_b)

class LineCaster(object):
    """Associate a line with a pointcloud, and provides methods for fitting 
    a line to its pointcloud. """

    def __init__(self, pointcloud, centerpoint=None):
        self.pointcloud = pointcloud
        if centerpoint is None:
            centerpoint = choice(pointcloud)
        self.centerpoint = centerpoint
        
        #quickly fit the line to the passed pointcloud
        self._try_pointing_at_every_point()

    def _try_pointing_at_every_point(self, min_theta_resolution=.05):
        """Local search method of collection of angles that point at 
        points in the pointcloud. """
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

class LCPopulation(object):
    """Store a collection of linecasters, and provide methods 
    to manipulate and select subsets of the stored linecasters"""
    def __init__(self, points, n_lines, local_neighborhood_radius=.4):
        """points should be a list of points.

        n_lines determines how many lines this population should contain."""
        self.pointcloud = Pointcloud(points, local_neighborhood_radius) 
        self.linecasters = []
        self.local_neighborhood_radius = local_neighborhood_radius
        

        #now fill in the population of optimized lines
        for _ in range(n_lines):
            #select a point the line will pass through
            point = choice(points)
            
            #get the points close to this point 
            close_points = self.pointcloud.query_ball_point(point)
            
            self.linecasters.append(LineCaster(close_points, point))

    def calc_best_order(self, n_lines):
        """really really slow do not use
        
        Find the ordered subset of the collection of lines that gives the greatest 
        total of the squares of  the scores of the selected n lines. """
        #this is where a metaheuristic should be I guess 
        #lets see how well a simple heuristic works first 
        selected_LCs = []
        
        #we need a copy of the pointcloud we can strip data from 
        pointcloud = deepcopy(self.pointcloud)
        lcs = deepcopy(self.linecasters) #and these 
        
        for _ in range(n_lines):
            #check that every point hasn't been explained
            if not len(pointcloud.points):
                break 
                
            #update scores for new pointcloud
            self._rescore(lcs, pointcloud)
            
            #obtain best lc
            lcs.sort(key=lambda lc: lc.score)
            selected_lc = lcs[-1]
            selected_LCs.append(selected_lc)
            del lcs[-1]
            
            #pluck the explained points from the population
            _explained, unexplained = pointcloud.segment_over_line(selected_lc.line)
            pointcloud = Pointcloud(unexplained, self.local_neighborhood_radius)
            
        return selected_LCs
            
            
    def _rescore(self, linecasters, pointcloud):
        """update the score of linecaster to use the passed 
        pointcloud, but do not reassociate the linecasters to 
        the pointcloud. This will unsync the linecaster score 
        attribute from its intended true value. """
        for lc in linecasters:
            lc.line.score(pointcloud.points)
            lc.score = lc.line.most_recent_score