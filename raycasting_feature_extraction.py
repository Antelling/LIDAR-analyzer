"""ALGORITHM STEPS: 
1. Raycasting procedure
    1.1 Select a point P.
    1.2 select a theta value to maximize the sum of subfunction:
        1.2.1 Cast a line passing through P with slope theta.
        1.2.2 For every point in the pointcloud, record the 
            probability that the point would be caused by 
            line(P, theta).
        1.2.3 Return the probabilties in a vertical column
    1.3 return result of subfunction call with optimized theta value
2. Construct matrix line_point_incidences made up of 
    200 (?) columns given by step 1 
3. Column Difference Rank procedure
    3.1 With two columns a and b,
    3.2 Let differences = [|a[i] - b[i]| for i in range(len(a))]
    3.3 Return sum(differences)
4. Construct a new 200x200 symmetrical matrix lines_to_lines 
    One column represents one lines relationship to every other 
    line. A small value indicates a high degree of similarity.
    4.1 Take the 100 (?) lowest values from the matrix. 
    4.2 Rank every column according to how many of the 100 lowest 
        values it contains.
    4.3 Select the highest ranking column to add to the set of selected 
        lines. Remove all columns from matrix that are too similiar (how)
        to the selected line/column.
    4.4 Do step 4.3 until some stopping criteria is met. 
"""

import numpy as np
from random import choice as rand_choice
import line_fit
from scipy import spatial 

class RayCaster(object):
    def __init__(self, pointcloud, centerpoint=None):
        self.pointcloud = pointcloud
        if centerpoint is None:
            centerpoint = rand_choice(pointcloud)
        self.centerpoint = centerpoint
        self.try_pointing_at_every_point()

    def try_pointing_at_every_point(self, min_theta_resolution=.05):
        #because we did the kmeans centroid clustering, 
        #a lot of our points are in straight lines. 
        #walls will pass through all those points, 
        #so when searching for the best theta we can 
        #set theta based on other points. 
        tried_thetas = []
        best_found_theta = 0
        best_found_score = 0

        self.line = line_fit.Line(self.centerpoint[0], self.centerpoint[1], 0)

        for point in self.pointcloud:
            if np.allclose(point, self.centerpoint):
                continue 
            theta = np.arctan2(point[1] - self.centerpoint[1], point[0] - self.centerpoint[0])
            if tried_thetas and min([np.abs(t-theta) for t in tried_thetas]) < min_theta_resolution:
                continue 
            tried_thetas.append(theta)
            self.line.params["theta"] = theta 
            score = self.line.score(self.pointcloud)
            if score > best_found_score:
                best_found_score = score 
                best_found_theta = theta 

        self.line.params["theta"] = best_found_theta

    def score(self, pc):
        self.line.score(pc)
        self.most_recent_score = self.line.most_recent_score


    def local_search_optimize_theta(self):
        return self._region_search(0, np.pi, 20)

        #first, rotate around with a big step size 
        # lower, upper = self._step_until_local_minima_found(.1)
        #reset theta to the lower value then run a more detailed search 
        # self.line.params["theta"] = lower 
        # lower, upper = self._step_until_local_minima_found(.05)
        #now run the region search 
        # self._region_search(lower, upper, 10)

    def _step_until_local_minima_found(self, step):
        """ spin theta around until spinning it any more or less would 
        decrease the score. Set the raycasters line to have this theta, 
        and then return the lower and upper theta values. """
        prev_line = line_fit.Line(self.line.params["x"], self.line.params["y"], self.line.params["theta"] - step)
        prev_line.score(self.pointcloud)

        next_line = line_fit.Line(self.line.params["x"], self.line.params["y"], self.line.params["theta"] + step)
        next_line.score(self.pointcloud)

        #we are rotating the line, and keeping track of how good it used to be, and how
        #good it is about to be. When how good it currently is is better than how it was previously 
        #and after this, then there is a local minima contained in between previous and next
        #however, we don't want to spin around forever. That would be rare but very bad
        total_rotation = 0
        while np.abs(total_rotation) < np.pi * 2:
            if self.line.most_recent_score > prev_line.most_recent_score and self.line.most_recent_score > next_line.most_recent_score:
                #we have found a bound for the local minima 
                return prev_line.params["theta"], next_line.params["theta"]
            else:
                total_rotation += step 
                prev_line = self.line 
                self.line = next_line 
                next_line = line_fit.Line(self.line.params["x"], self.line.params["y"], self.line.params["theta"] + step)
                next_line.score(self.pointcloud)

        #we failed to find a local minima so theta doesn't matter 
        return prev_line.params["theta"], next_line.params["theta"]

    def _region_search(self, theta_lower_bound, theta_upper_bound, n_steps):
        """ test everything in the region  """
        #take the initial line state as the best found score 
        best_found_theta = self.line.params["theta"]
        best_found_score = self.line.most_recent_score

        #how big should the steps be
        step_size = (theta_upper_bound - theta_lower_bound)/n_steps

        #set the line to the start location 
        self.line.params["theta"] = theta_lower_bound

        #test everything
        for _ in range(n_steps + 1):
            self.line.score(self.pointcloud)
            if self.line.most_recent_score > best_found_score:
                best_found_theta = self.line.params["theta"]
                best_found_score = self.line.most_recent_score
            self.line.params["theta"] += step_size

        self.line.params["theta"] = best_found_theta 

        
class RFE(object):
    def __init__(self, points, n_lines):
        """Fit n_lines to the points"""
        kdtree = spatial.KDTree(points)
        self.points = points 
        self.kdtree = kdtree 
        self.lines = line_fit.LinePopulation()
        for _ in range(n_lines):
            #select a random point 
            point = rand_choice(points)
            #get the points close to this point 
            close_point_indexes = kdtree.query_ball_point([point], .4)[0]
            close_points = [points[i] for i in close_point_indexes]
            #init and fit a raycaster
            rc = RayCaster(close_points, point)
            self.lines.append(rc)

    def remove_duplicate_lines(self, cartesian_scale=1, polar_scale=1, threshold=.2):
        """when we fit lines, we consider only their local points. However, 
        walls, or at least the lines that walls fall along, might go on for 
        much longer. Here, we rescore every line to use the full collection 
        of points, then sort them by their score. Lines are put into a new 
        collection that does not allow duplicates in the order of their score."""
        self.lines.rescore_solutions(self.points)
        self.lines.sort()
        line_set = line_fit.LinePopulation()
        for rc in self.lines.population:
            already_in = False 
            for otherrc in line_set.population:
                if rc.line.is_similar(otherrc.line):
                    #we already have this wall recorded 
                    already_in = True 
                    break 
            if not already_in:
                line_set.append(rc)
        self.lines = line_set 

    







