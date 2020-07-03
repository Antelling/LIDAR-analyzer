using ForwardDiff
using Plots, Random
using LinearAlgebra: normalize
using QuadGK

const Point = Vector{Float64}
const Points = Array{Float64,2}
const Params = Any

const line_points = [
    1 1
    2 2
    3 3
    4 4.0
]

function least_squares_score(points::Points, params::Params;)
    num = params[2] - params[4]
    den = params[1] - params[3]
    m = (params[2] - params[4]) / (params[1] - params[3])
    b = params[2] - (m * params[1])

    total_error = 0.0
    for point in points
        total_error += (m * point[1] + b - point[2])^2
    end

    -total_error
end

function score(
    points::Points,
    params::Params;
    scaler::Float64 = 0.5,
    flattener::Float64 = 0.2,
)
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
            1.4 add attraction to the score
        2. return the score squared divided by the line length"""
    #calculate slope and y intercept
    m = (params[2] - params[4]) / (params[1] - params[3])
    b = params[2] - (m * params[1])

    #calculate segment length
    l = max(1, hypot(params[1] - params[3], params[2] - params[4]))

    #sum over every point
    total_attraction = 0.0
    for point in eachrow(points)

        #calculate attraction
        closest_val = (-b * m + m * point[2] + point[1]) / (m^2 + 1)
        closest_val = max(min(params[1], params[3]), closest_val)
        closest_val = min(max(params[1], params[3]), closest_val)
        orth_distance =
            hypot(point[1] - closest_val, point[2] - closest_val * m + b)
        total_attraction += 1 / (flattener + (orth_distance * scaler)^2)
    end

    total_attraction^2 / l
end



function point_heat_calc(m, b, l, h, x, y, s)
    """Calculate total heat applied to this line by the passed point.
    Parameters:
        m - slope of the line
        b - y-intercept of the line
        l - lowest x-value of the line
        h - highest x-value of the line
        x - x value of the point heat source
        y - y value of the point heat source
        s - smooth factor - added to squared distance when computing the inverse
            square value: 1/(s + d^2)
    """
    # Heat on a point is given by the inverse square law applied
    # to orthogonal distance from the point on the line to the heat point
    # source. This function should therefore express the integral of
    # 1/(s+(sqrt((v-x)^2+((m*v+b)-y)^2)^2) from l to h wrt v.
    # The computed integral is unfortunately gigantic.
    sqrt_part = sqrt(-(y^2) + (2*m*x + 2*b)*y - (m^2)*x^2 - 2*b*m*x + (-(m^2) - 1)*s - (b^2))
    ln_sqrt_part = 2*sqrt(-(y^2) + (2*m*x+2*b)*y - m^2*x^2 - 2*b*m*x + (-(m^2)-1)*s-(b^2))
    ln_other_part(term) = (2*m*y + 2*x + -2*term*m^2 + -2*b*m + -2*term)
    ln_part(term) = log(abs(ln_sqrt_part+ln_other_part(term))/abs(ln_sqrt_part-ln_other_part(term)))
    bottom_part = 2(y^2 + (-2*m*x - 2*b)*y + m^2*x^2 + 2*b*m*x + (m^2+1)*s+b^2)
    return (sqrt_part*(ln_part(l) - ln_part(h)))/bottom_part
    #see https://www.integral-calculator.com/ with 1/(s+(sqrt((v-x)^2+((m*v+b)-y)^2)^2))
    1/(s+(sqrt((v-x)^2+((m*v+b)-y)^2)^2))
end

function point_heat_calc2(m, b, l, h, x, y, s)
    """how much heat will this point apply to the passed line"""
    denom_part = sqrt(y^2+(-2*m*x - 2*b)*y + m^2*x^2 + 2*b*m*x + (m^2+1)*s + b^2)
    atan_part(dim) = atan((m*y + x - dim*m^2 - b*m - dim)/denom_part)
    atan_part(l)/denom_part - atan_part(h)/denom_part
end

function point_heat_calc3(m, b, l, h, x, y, s)
    denom_part = sqrt(y^2+(-2*m*x - 2*b)*y + m^2*x^2 + 2*b*m*x + (m^2+1)*s + b^2)
    atan_part(dim) = atan((m*y + x - dim*m^2 - b*m - dim)/denom_part)
    abs(m)*(atan_part(l)/denom_part - atan_part(h)/denom_part)
end

point_heat_calc3(1, 0, 0, 5, 1, 0, .2)
point_heat_calc2(1, 0, 0, 5, 1, 0, .2)
point_heat_calc3(1, -.5, 0, 5, 1, 0, .2)
point_heat_calc3(0, 0, -100, 100, 0, 0, .2)
point_heat_calc2(0, 0, -100, 100, 0, 0, .2)
point_heat_calc3(100, 0, -100, 100, 0, 0, .2)

function heat_score(points::Points, params::Params; smoother::Float64=.5)
    """attempt to maximize the total temperature of the line, treating
    each point as a heat source.

    In order to
    """
    if params[1] < params[3]
        low_x = params[1]
        low_y = params[2]
        high_x = params[3]
        high_y = params[4]
    else
        low_x = params[3]
        low_y = params[4]
        high_x = params[1]
        high_y = params[2]
    end
    m = (high_y - low_y) / (high_x - low_x)
    b = low_y - (m * low_x)
    l = hypot(low_x - high_x, low_y - high_y)

    heat = sum([point_heat_calc3(m, b, low_x, high_x, px, py, smoother) for (px, py) in eachrow(points)])
    return heat^2/l
end


function scorer_closure(score_func, points::Points, kwargs::Dict)
    """Parameters cannot be passed through ForwardDiff. This
    method returns a score instance associated with the passed points and
    scaler value."""
    return function scorer_closure_internal(params)
        return score_func(points, params; kwargs...)
    end
end

function optimize(
    p,
    scorer;
    delta = 0.05,
    allowable_failures::Int = 5,
    friction = 0.99,
)
    best_found = [copy(p), scorer(p)]
    prev_score = -1
    score = 0
    momentum = zeros(length(p))
    failures = 0
    while failures <= allowable_failures
        vector = ForwardDiff.gradient(scorer, p)
        momentum += delta .* vector
        momentum *= friction
        p += momentum
        prev_score = score
        score = scorer(p)
        if score > best_found[2]
            best_found = [copy(p), score]
            failures = 0
        else
            failures += 1
        end
    end
    best_found[1]
end

function graph(points, params, score)
    x = points[:, 1]
    y = points[:, 2]
    scatter(x, y, title = "Score: $(score)")
    plot!([params[1], params[3]], [params[2], params[4]])
end

function random_param()
    return (rand()) * 100
end

function test(n_trials = 100; n_dimensions = 4, delta = 0.05)
    scorer = scorer_closure(points, 0.5, 0.5)

    best_found = [0, []]
    for trial_i = 1:n_trials
        params = [random_param() for _ = 1:n_dimensions]
        println("random score is $(scorer(params))")
        optimize(params, scorer, delta = delta)
        println("large step score is $(scorer(params))")
        optimize(params, scorer, delta = delta / 4)
        println("small step score is $(scorer(params))")
        if scorer(params) > best_found[1]
            best_found = [scorer(params), params]
        end
    end

    best_found
end

# test(1, delta=10)
# bf = test(100, delta=1)
# graph(points, bf[2])

using PyCall
push!(pyimport("sys")."path", pwd());
pointcloud = pyimport("pointcloud")
dataloader = pyimport("data_handler")
dl = dataloader.DataLoader("data_2020-06-10-10-24-18.bag")
pc = pointcloud.Pointcloud(dl.load_next_frame())
pc.remove_floor(floor = 0.05)
pc.take_xy()
pc.take_percentage(0.5)
pc.biased_undersample(percentile = 0.1, radius = 0.6)
pc.take_centroids(400, exact = true)
rpoints = pc.points

scorer = scorer_closure(heat_score, line_points, Dict())

# params = [random_param() for _ in 1:4]
params = [1.5, 1.8, 4.5, 5.3]
params = optimize(params, scorer, delta = 1.0, allowable_failures = 1)
params = optimize(params, scorer, delta = 0.00001, allowable_failures = 200)
graph(line_points, params, scorer(params))


scorer(params)
