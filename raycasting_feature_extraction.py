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
import line_fit

class RayCaster(object):
    def __init__(self, pointcloud):
        self.pointcloud = pointcloud
        centerpoint = np.random.choice(pointcloud)
        theta = np.random.uniform(np.pi * 2)
        self.line = line_fit.Line(x=centerpoint[0], y=centerpoint[1], theta=theta)

    def theta_local_search(self, step, convergence=.01):
        """  apply step to the theta until the score goes from increasing to decreasing.
        
        Then apply a logistic search until the upper and lower bounds for the location of 
        the local minima for the theta value is pinned down within the convergence. """
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
                return self._logistic_search(prev_line.params["theta"], next_line.params["theta"])
            else:
                total_rotation += step 

    def _logistic_search(self, bound_a, bound_b):
        pass

