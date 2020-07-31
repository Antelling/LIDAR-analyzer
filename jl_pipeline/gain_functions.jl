module GainFunctions
"""Traditionally, linear optimization is concerned with reducing a loss
function. Least Squares Error attempts to minimize the squared prediction error.
Here, we are attempting to maximize a gain function. We want lines to explain
the maximum amount of points. We don't know what the ideal gain function is,
or what the best parameters are."""

include("hyperparam_spaces.jl")

struct GainFunction
    """Each formula accepts a distance and
    a list of parameters. The parameters have
    different meanings for different formulas,
    but are always floats. """
    formula::Function
    param_space::PS.ParamSpaces
    params::Dict
end
function GainFunction(formula, param_space::PS.ParamSpaces)
    params = PS.sample(param_space)
    GainFunction(formula, param_space, params)
end

function resample(gf::GainFunction)
    new_params = PS.sample(gf.param_space)
    GainFunction(gf.formula, gf.param_space, new_params)
end

function inverse_exponent(distance; exponent=2, scaler=1, smoother=.1)
    smoother/(smoother + (distance*scaler)^exponent)
end

function threshold_distance(distance; scaler=1)
    max(1 - (distance*scaler), 0)
end

function gaussian_pdf(distance; std=.5)
    exp((-(distance)^2)/(2*std^2))/(std*sqrt(2pi))
end

function eval(gain_func::GainFunction, d)
    gain_func.formula(d; gain_func.params...)
end

function graph(gain_func::GainFunction; l=0, h=5, overwrite=true)
    gf(d) = eval(gain_func, d)
    if overwrite
        plot(gf, l, h)
    else
        plot!(gf, l, h)
    end
end

function value_closest_to_point(m, b, x, y)
    """Find the x value where the line described by m and
    b passes closest to the (x, y) point """
    (-b * m + m * y + x) / (m^2 + 1)
end

function score(m, b, l, h, points, gain_func)
    """Score the passed line with the passed points according
    to the gain function. """
    tot = 0
    for point in eachrow(points)
        v = value_closest_to_point(m, b, point[1], point[2])
        v = max(l, v)
        v = min(h, v)
        d = hypot(point[1] - v, point[2] - (m*v+b))
        tot += eval(gain_func, d)
    end
    tot
end

inv_exp_params = PS.ParamSpaces(
    :exponent=> PS.Uniform(1, 5),
    :scaler=> PS.Switch([PS.Uniform(0, 1), PS.Uniform(0, 10)]),
    :smoother=> PS.Switch([PS.Uniform(0, .3), PS.Uniform(0, 1.5)])
)
thresh_dist_params = PS.ParamSpaces(
    :scaler=> PS.Uniform(0, 2)
)
gauss_pdf_params = PS.ParamSpaces(
    :std=> PS.Uniform(0, 2)
)

gain_functions = [
    GainFunction(inverse_exponent, inv_exp_params),
    GainFunction(threshold_distance, thresh_dist_params),
    GainFunction(gaussian_pdf, gauss_pdf_params)
]
export GainFunction, gain_functions
end
