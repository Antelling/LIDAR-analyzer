import numpy as np
import random, math, copy
import scipy.odr
from scipy.stats import norm
import sensor_msgs.point_cloud2

class Line(object):
    """This method represents a line and provides methods of doing math involving other lines. """

    def __init__(self, x: float, y: float, theta:float):
        """accept x, y, and theta, and store them under a self.params dictionary. """
        self.params = {"x": x, "y": y, "theta": theta}

    def score(self, pointcloud: sensor_msgs.point_cloud2):
        """we want score to reflect the summation of the likelihood that a wall occupying this space would trigger a recorded point for each point in the pointcloud.
        This method is intended to reflect the amount of points that are explained by a wall occupying this position. """
        #FIXME: this is really slow. Possible optimizations:
        #use numpy operations instead of python
        #use a spatial datastructure to store the points instead 
        #   of a flat array (what kind/how would it be faster)
        total = 0
        for point in pointcloud:
            chance = self._likelihood_of_being_cause_of_point(point)
            #should the chance map exactly to the score we're trying to optimize for? should it be a threshold? 
            total += chance 
        self.most_recent_score = total
        return total 

    def _likelihood_of_being_cause_of_point(self, point):
        #LIDAR scans in cylindrical dimensions. This method should convert the cartesian 
        # point to a cylindrical one and then use the standard deviation along each dimension 
        # to determine the likelihood the wall described by this Line object would cause the 
        # passed point. 
        #FIXME: I don't know how to do the math for this. I'm going to take shortest cartesian 
        # distance then run it through a gaussian \
        x0, y0 = point[0], point[1]
        x1, y1 = (self.params["x"], self.params["y"])
        x2, y2 = (self.params["x"] + np.cos(self.params["theta"]), self.params["y"] + np.sin(self.params["theta"]))
        distance = np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)/np.sqrt((y2-y1)**2 + (x2-x1)**2)
        if math.isnan(distance):
            distance = 100
        chance = 1 - norm.cdf(distance*2)
        return chance   

    def jaya_creation(self, top_solution, bottom_solution):
        #the jaya algorithm involves applying a uniform 0-100% of the vector from this solutions 
        #location to the top solutions location, and then another uniform 0-100% vector from the worst solution to the 
        #startion location of this solution. this method returns a new Line instance
        top_vector = top_solution.params_subtract(self.params)
        bottom_vector = self.params_subtract(self.params)
        new_line = self.apply_params(top_vector, scale=np.random.rand()).apply_params(bottom_vector, scale=np.random.rand())
        return new_line

    def params_subtract(self, params):
        """return a param dict of the self location minus the passed params"""
        return {
            "x": self.params["x"] - params["x"],
            "y": self.params["y"] - params["y"], 
            "theta": self._theta_subtract(params["theta"])
        }

    def _theta_subtract(self, theta):
        """rotate one theta until its at the closest position to the other, then do the subtraction. """
        thetas  = sorted([self.params["theta"], theta])
        while thetas[0] <= thetas[1]:
            prev_theta = thetas[0]
            thetas[0] += np.pi * 2
        difference_a = prev_theta - thetas[1]
        difference_b = thetas[0] - thetas[1]
        return min(difference_a, difference_b)

    def apply_params(self, vector, scale=1.0):
        """Return a new Line instance based off the current parameters plus the passed vector times the passed 
        scaling factor. """
        vector = self._scale_vector(vector, scale)
        return Line(x=self.params["x"] - vector["x"], y=self.params["y"] - vector["x"], theta=self._theta_subtract(vector["theta"]))

    def _scale_vector(self, vector, scale):
        """accept a dict of params and multiplies each param by the scale"""
        return {key: value * scale for (key, value) in vector.items()}

    def is_similar(self, 
            other_line, 
            cartesian_tolerance_scale=1, 
            theta_tolerance_scale=.5, 
            tolerance=1):
        """Return if the passed line is too similar to this line"""
        distance = np.sqrt(
            (self.params["x"] - other_line.params["x"])**2 + 
            (self.params["y"] - other_line.params["y"])**2 ) 
        adjustments = (np.cos(self.params["theta"]) * distance, np.sin(self.params["theta"]) * distance)

        for sign in [1, -1]:
            new_line = Line(
                self.params["x"] + (sign * adjustments[0]),
                self.params["y"] + (sign * adjustments[1]),
                self.params["theta"])
            #get the total of the differences of parameters between this new 
            #representation of the first line and the second line
            total = np.abs(new_line.params["x"] - other_line.params["x"]) * cartesian_tolerance_scale
            total += np.abs(new_line.params["y"] - other_line.params["y"]) * cartesian_tolerance_scale
            total += np.abs(new_line._theta_subtract(other_line.params["theta"])) * theta_tolerance_scale

            if total < tolerance:
                return True
        return False


    def split_up_pointcloud(self, pointcloud, threshold=.75):
        """return two pointclouds - the first containing points likely 
        explained by self line, and the second containing the unexplained points."""
        explained = []
        unexplained = []
        for point in pointcloud:
            lik = self._likelihood_of_being_cause_of_point(point)
            if lik > threshold:
                explained.append(point)
            else:
                unexplained.append(point)
        return explained, unexplained 


class LinePopulation(object):
    def __init__(self, population=None):
        if population is None:
            self.population = []
        else:
            self.population = population 

    def add_n_solutions(self, n, scale=4):
        for _ in range(n):
            x = (np.random.uniform() - .5) * scale,
            y = (np.random.uniform() - .5) * scale,
            theta = np.random.uniform() * np.pi * 2
            line = Line(x[0], y[0], theta) #python is making x and y tuples 
            self.population.append(line)

    def add_passed_solutions(self, solutions):
        self.population += solutions

    def _already_in_population(self, line):
        for otherline in self.population:
            if line.is_similar(otherline):
                return True 
        return False

    def take_top_solution(self):
        """remove the top scoring population, according to the most
        recently evaluated scores, from the population, then return 
        it."""
        self.population.sort(key=lambda line: line.most_recent_score)
        top = self.population[-1]
        del self.population[-1]
        return top

    def rescore_solutions(self, pointcloud):
        """update the most recent score for every solution in this
        set to be accurate wrt the passed pointcloud"""
        for i in range(len(self.population)):
            self.population[i].score(pointcloud)

    def append(self, line):
        self.population.append(line)

    def sort(self):
        #sort the lines in descending order of score 
        self.population.sort(key=lambda x: -x.most_recent_score)
            

class JayaPopulation(LinePopulation, object):
    """A LinePopulation that also has methods for the jaya metaheuristic. """

    def apply_iteration(self, pointcloud, n=4):
        """apply the jaya transformation once to each line in the population
        
        the parameter n controls how many of the top and bottom solutions are to 
        be selected from. A value of 1 is used in Dr. Rao's formulations, but higher
        values are much more effective.  """
        self.population.sort(key=lambda line: line.most_recent_score)
        top_sol = random.choice(self.population[-n:-1])
        bottom_sol = random.choice(self.population[0:n])

        imp = False

        top_sol
        #now we loop over every solution and mutate it 
        for i in range(len(self.population)):
            new_sol = self.population[i].jaya_creation(top_sol, bottom_sol)
            if new_sol.score(pointcloud) > self.population[i].most_recent_score:
                self.population[i] = new_sol
                imp = True
        return imp

    def run_until_n_consecutive_fails(self, n_fails, pointcloud):
        """apply jaya iterations repeatedly until an improvement fails to 
        be found n_fails in a row"""
        current_fails = 0
        while current_fails <= n_fails:
            imp = self.apply_iteration(pointcloud)
            if imp:
                current_fails = 0
            else:
                current_fails += 1


class WallLocatorJaya(object):
    """This wall locator uses the jaya metaheuristic to fit lines to passed pointclouds. 
    Consecutive pointclouds will not change much, since the robot is pretty slow, so although
    fitting the lines initially is expensive, updating lines for new pointcloud data should 
    converge faster. """
    def __init__(self, jaya_popsize=40):
        self.population = JayaPopulation()
        self.population.add_n_solutions(jaya_popsize)
        self.jaya_popsize = jaya_popsize


    def fit_walls(self, pointcloud, n_walls):
        """Accept a pointcloud object, and determine the best algorithm to 
        use to fit walls to it. """
        return self._jaya_fit_walls(n_walls, pointcloud)

    def _jaya_fit_walls(
            self, 
            n_walls,
            pointcloud, 
            initial_permitted_consecutive_failures=15,
            updates_permitted_consecutive_failures=5,
            take_every_time=3):
        """Use the jaya population-based metaheuristic to fit a population of 
        potential solutions to a pointcloud."""
        #each solution has an associated score, which is meant to represent how many 
        # points are covered by the solution. We do not want to double count points, 
        # so after selecting a solution to be included in our set of walls, we should 
        # remove the points covered by that wall from the solution. Or should points 
        # not be binary, and instead we weight them based on the chance they have previously
        # been explained by a wall in our set of walls?

        #first, update our population of good solutions to reflect the values of the new 
        #pointcloud 
        self.population.rescore_solutions(pointcloud)

        #apply jaya to update our current population of lines to this new pointcloud
        self.population.run_until_n_consecutive_fails(initial_permitted_consecutive_failures, pointcloud)

        #now, repeat taking walls from our population until we have as many as requested 
        selected_walls = []
        while len(selected_walls) < n_walls:
            for _ in range(take_every_time):
                chosen_solution = self.population.take_top_solution()
                #after we select a wall, we should remove all points from the pointcloud
                #that are likely explained by the wall 
                explained_points, unexplained_points = chosen_solution.split_up_pointcloud(pointcloud)
                #store the selected line and its associated points
                selected_walls.append([chosen_solution, explained_points])
                #replace the current pointcloud with the unexplained points 
                pointcloud = unexplained_points
                #update the scores of the line population for the new pointcloud 
                self.population.rescore_solutions(pointcloud)
                #optimize the new, smaller population of lines for the reduced pointcloud
                self.population.run_until_n_consecutive_fails(updates_permitted_consecutive_failures, pointcloud)
                #the above line of code runs one more time than it needs to, on the last loop
        
        #we now have an object holding n_walls walls and their associated points, 
        # a population of lines that were not very good at explaining anything, 
        # and a collection of points that were not explained by any of the chosen lines. 
        #we want to combine the set of chosen lines back with the set of unchosen lines, 
        #so future runs of this technique will be able to draw from an already well-optimized
        #set of solutions. 
        chosen_lines = [line for (line, points) in selected_walls]
        self.population.add_passed_solutions(chosen_lines)

        #now we can fit each line to its pointcloud, or we can 
        # pass the collection of lines to the pointcloud optimizer, which 
        # will consider all lines at once when assigning points from the 
        # combined pointcloud to lines, and then fit each line on its assigned points. 
        # Or we could just return the lines as they are.
        #the trust region approach is probably the most correct but I don't have 
        # that implemented yet, so: 
        return chosen_lines

    def _trust_region_update_walls(self, n_walls, pointcloud):
        """once we have lines fit, we can get a more precise fit 
        by assigning each point in the current pointcloud to its
        most likely explanatory line, and then fitting each line 
        to its points. """
        print("trust region update is not implemented. ")
        return self._jaya_fit_walls(n_walls, pointcloud)

    def _clean_pointcloud(self, pointcloud, percent=.1):
        """take the passed percent of points"""
        #FIXME: add a denoising filter, probably? does the lidar module do that already
        pointcloud = [point for point in pointcloud.population if np.random.rand()<percent]