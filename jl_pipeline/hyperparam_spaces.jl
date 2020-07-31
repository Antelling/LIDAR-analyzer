"""A parameter is one of the coordinates in the line.
A hyperparameter is a parameter that controls the optimization
of those parameters.
We are interested in finding hyperparameters that result in
good parameters being quickly found.
A hyperparameter can be any type or magnitude.
Here, we provide probability distributions to sample from to get
a set of hyperparameters. """

module PS

abstract type ParamSpace end

struct Uniform <: ParamSpace
    low::Number
    high::Number
end

struct IntRange <: ParamSpace
    range::UnitRange
end

struct Switch <: ParamSpace
    options::Vector{ParamSpace}
end

const ParamSpaces = Dict{Symbol,ParamSpace}
function sample(ps::ParamSpaces)
    Dict(key => sample(val) for (key, val) in ps)
end

function sample(uni::Uniform)
    uni.low + rand() * (uni.high - uni.low)
end

function sample(swi::Switch)
    sample(rand(swi.options))
end

function sample(ints::IntRange)
    rand(ints.range)
end
# EvoStatPS = ParamSpaces(
#     "n_children" => Switch(
#         [Uniform(1, 10), Uniform(10, 100), Uniform(100, 1e3),
#         Uniform(1e3, 1e4), Uniform(1e4, 1e5), Uniform(1e5, 1e6)]
#     ),
#     "range" => Uniform(0, 2)
# )
#
# sample(EvoStatPS)

export Uniform, Switch, ParamSpaces, sample
end
