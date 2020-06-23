import numpy as np 
from matplotlib import pyplot as plt 
from line import LCPopulation


gas = GASelector(lf, Pointcloud(good_points, .07), popsize=30, lines_in_sol=25)
for x in range(60):
    gas.run_iter()
best_solution = gas.best_solution()
graph(good_points)
for line in best_solution["lines"]:
    lf.linecasters[line].line.score(good_points)
    pc = Pointcloud(good_points, .06)
    explained, good_points = pc.segment_over_line(lf.linecasters[line].line)
    graph_line(lf.linecasters[line].line, plt)
plt.show()
best_solution
