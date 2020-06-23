from random import randint, choice, choices 
from line import Line, line_point_distance
import numpy as np

class GASelector(object):
    def __init__(self, linecasters, pointcloud, popsize=100, lines_in_sol=10, local_neighborhood_radius=.04):
        """Set up a genetic algorithm metaheuristic"""
        #one solution is an ordered list of indexes of lines
        self.solutions = []
        self.linecasters = linecasters 
        self.pointcloud = pointcloud 
        self.local_neighborhood_radius = local_neighborhood_radius

        self._make_line_point_distance_matrix()

        #init some randomly selected groups of solutions
        for _ in range(popsize):
            line_indexes = [randint(0, len(self.linecasters.linecasters)-1) for _ in range(lines_in_sol)]
            self.solutions.append(self._make_solution(line_indexes))

    def _make_line_point_distance_matrix(self):
        """Calculate the distance between every line and every point"""
        #first, compute the distance matrix between each line and each point 
        matrix = []
        for lc in self.linecasters.linecasters:
            distances = [line_point_distance(lc.line, point) for point in self.pointcloud.points]
            matrix.append(distances)
        self.matrix = matrix


    def _make_solution(self, line_indexes):
        """associate an ordered collection of lines with a score"""
        return {
            "lines": line_indexes,
            "score": self._score(line_indexes)
        }

    def _score(self, line_indexes):
        """determine the score of a list of line indexes"""
        explained = np.zeros(len(self.pointcloud.points))
        score = 0
        for line_index in line_indexes:
            for (point_index, point_index_explained) in enumerate(explained):
                #check that this point hasn't already been explained
                if point_index_explained:
                    continue 

                #check if this point is being explained 
                if self.matrix[line_index][point_index] < self.local_neighborhood_radius:
                    explained[point_index] = 1 #disable this point
                    score += 1
        return score 
                
                
    def run_iter(self):
        """loop through every solution, select another solution, generate a child, 
        then replace the lowest performing solution's parent if the child outperforms it.  """
        for (solution_index, solution) in enumerate(self.solutions):
            #select another solution that is not this one 
            other_solution_index = solution_index 
            while other_solution_index == solution_index:
                other_solution_index = randint(0, len(self.solutions) - 1)
            
            #combine lines from both parents 
            gene_pool = list(set(self.solutions[other_solution_index]["lines"]) | set(solution["lines"]))
            #generate child 
            child = choices(gene_pool, k=len(solution["lines"]))
            if len(child) < len(solution):
                continue
            #mutate gene 
            child[4] = max(child[4] - 1, 0)
            child = self._make_solution(child)

            #compare child to parents
            if child["score"] > solution["score"]:
                self.solutions[solution_index] = child
            elif child["score"] > self.solutions[other_solution_index]["score"]:
                self.solutions[other_solution_index] = child 

    def best_solution(self):
        return max(self.solutions, key=lambda x: x["score"])