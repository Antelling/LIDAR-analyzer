const Point = Vector{Float64}
const Points = Vector{Point}
const Params = Any
using ForwardDiff
using Plots

const points = [[89.1, 39], [98, 94], [76, 54], [9, 45], [49, 20], [20, 27],
[81, 67], [45, 83], [6, 15], [43, 89], [24, 71], [23, 37], [38, 82], [8, 16],
[56, 46], [71, 50], [42, 86], [30, 22], [47, 69], [3, 62], [4, 28], [9, 77],
[25, 74], [2, 31], [84, 69], [95, 78], [16, 89], [36, 34], [89, 52], [58, 30],
[64, 2], [70, 38], [1, 17], [75, 38], [69, 0], [24, 64], [57, 11], [71, 54],
[49, 5], [76, 21], [57, 20], [52, 98], [2, 72], [23, 13], [83, 2], [97, 45],
[53, 59], [6, 95], [80, 97], [27, 97], [11, 89], [15, 70], [10, 50], [61, 89],
[23, 34], [68, 90], [97, 61], [12, 80], [80, 81], [44, 46], [1, 24], [33, 42],
[11, 27], [14, 93], [76, 90], [15, 72], [57, 49], [58, 48], [11, 3], [96, 42],
[73, 60], [33, 20], [42, 70], [25, 22], [20, 39], [54, 57], [49, 93], [94, 44],
[49, 37], [78, 51], [67, 72], [19, 61], [20, 25], [51, 81], [31, 99], [24, 78],
[24, 19], [31, 26], [65, 45], [25, 92], [24, 59], [30, 5], [87, 6], [80, 63],
[92, 5], [53, 19], [97, 4], [30, 7], [8, 77], [8, 31]]

const test_points = [[2.0, 0.0], [3.0, 1.0]]

function score(
        points::Points,
        params::Params;
        scaler::Float64=.5,
        flattener::Float64=.2)
    """ Determine summed strength of the attraction to all points.
    Passed:
        points - Points object of input data
        params - Params object of x0, y0, x1, y1
        scaler - a float value to be multiplied by the shortest possible
            orthographic distance from each point to this line
        flattener - attraction is determined with the inverse square law, but
            points can get infinitely close to lines, so we add a flattener
            term to get 1/(flattener + (scaled_distance)^2)

    Algorithm Procedure:
        1. for each point in points:
            1.1 calculate the orthographic distance to the
            line segment described by params
            1.2 scale the orthographic distance by scaler
            1.3 let attraction = 1/(ortho_distance^2 + flattener)
            1.4 add attraction to the score"""
    #calculate slope and y intercept
    m = (params[1] - params[3]) / (params[2] - params[4])
    b =  params[2] - (m * params[1])

    #sum over every point
    total_attraction = 0
    for point in points

        #calculate attraction
        closest_val = (-b*m+m*params[2]+params[1])/(m^2+1)
        orth_distance = hypot(point[1] - closest_val,
                              point[2] - closest_val*m + b)
        total_attraction += 1/(flattener + (orth_distance*scaler)^2)
    end

    total_attraction
end

function scorer_closure(points::Points, scaler::Float64, flattener::Float64)
    """Parameters cannot be passed through ForwardDiff. This
    method returns a score instance associated with the passed points and
    scaler value."""
    return function scorer_closure_internal(params)
        return score(points, params, scaler=scaler, flattener=flattener)
    end
end

function optimize(p, scorer; delta=.05)
    for _ in 1:100
        vector = ForwardDiff.gradient(scorer, p)
        println("score is $(scorer(p))")
        p += delta .* vector
    end
    p
end

function graph(points, params)
    x = [p[1] for p in points]
    y = [p[2] for p in points]
    plot(x, y, series=:scatter)
    plot([params[1], params[3]], [params[2], params[4]])
end

function test()
    scorer = scorer_closure(points, .5, .5)
    p = [0, 0, 0, .2]
    results = optimize(p, scorer)
end

test()
