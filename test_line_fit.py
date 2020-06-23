from numpy import pi
import numpy as np

point_clouds = {
    "single line": [
        [1, 1, 1],
        [2, 2, 1],
        [3, 3, 1],
        [4, 4, 1],
        [5, 5, 1]
    ],
    "two lines": [
        [1, 1, 1],
        [2, 2, 1],
        [3, 3, 1],
        [4, 4, 1],
        [5, 5, 1],
        [1, 5, 1],
        [2, 5, 1], 
        [3, 5, 1], 
        [4, 5, 1],
        [5, 5, 1] #how should duplicate points be handled? 
    ],
    "three lines": [
        [1, 1, 1],
        [2, 2, 1],
        [3, 3, 1],
        [4, 4, 1],
        [5, 5, 1],
        [1, 5, 1],
        [2, 5, 1], 
        [3, 5, 1], 
        [4, 5, 1],
        [5, 5, 1],
        [10, 1, 1],
        [8, 2, 1],
        [6, 3, 1], 
        [4, 4, 1], 
        [2, 5, 1], 
        [0, 6, 1]
    ]
}

def test_Line():
    #check __init__ and score
    correct = line_fit.Line(0, 0, pi/4)
    incorrect = line_fit.Line(0, 0, pi/2)
    correct_score = correct.score(point_clouds["single line"])
    incorrect_score = incorrect.score(point_clouds["single line"])
    assert correct_score > incorrect_score

    #check that theta subtraction works
    #a half rotation minus a full rotation should give half a rotation
    #going left
    result = line_fit.Line(0, 0, pi)._theta_subtract(8*pi)
    assert result + pi < .001
    #a half rotation minus an eighth rotation should give 3/8 going right:
    result = line_fit.Line(0, 0, pi)._theta_subtract(pi/4)
    assert result - (3/8)*(2*pi) < .001

    #test is_similar works on two naively similar lines
    line_a = line_fit.Line(4, 4, pi/4)
    line_b = line_fit.Line(4.05, 4.05, pi/4 + .02)
    line_c = line_fit.Line(4, 4, pi/4 + pi/2)
    line_d = line_fit.Line(5, 5, pi/4)
    wacko = line_fit.Line(10, 40, -23.5)
    assert line_a.is_similar(line_d)
    assert line_a.is_similar(line_b)
    assert line_a.is_similar(line_c)
    assert line_b.is_similar(line_d)
    assert not line_a.is_similar(wacko)


def test_LinePopulation():
    pop = line_fit.LinePopulation()
    pop.add_n_solutions(20)
    assert len(pop.population) == 20

    second_pop = line_fit.LinePopulation()
    second_pop.add_n_solutions(11)
    pop.add_passed_solutions(second_pop.population)
    assert len(pop.population) == 31

def test_JayaPopulation():
    pop = line_fit.JayaPopulation()
    pop.add_n_solutions(30)
    assert len(pop.population) == 30
    pop.rescore_solutions(point_clouds["three lines"])
    pop.apply_iteration(point_clouds["three lines"])
    #idk how to test that it ran correctly


def test_WallLocatorJaya():
    locator = line_fit.WallLocatorJaya()
    walls = locator.fit_walls(point_clouds["three lines"], 3)
    print(walls)


test_Line()
test_LinePopulation()
test_JayaPopulation()
test_WallLocatorJaya()