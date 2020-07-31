module Optimizers
using ForwardDiff
include("coord_systems.jl")
include("hyperparam_spaces.jl")

struct Solution
    line::CoordSystems.Line
    score::Float64
end
Base.copy(s::Solution) = Solution(
    CoordSystems.Line(
        copy(s.line.coords),
        s.line.coord_system),
    s.score)

struct Optimizer
    optimizer::Function
    params::Dict
    params_spaces::PS.ParamSpaces
end
function Optimizer(optimizer::Function, params_spaces::PS.ParamSpaces)
    Optimizer(optimizer, PS.sample(params_spaces), params_spaces)
end

function resample(opt::Optimizer)
    new_params = PS.sample(opt.params_spaces)
    Optimizer(opt.optimizer, new_params, opt.params_spaces)
end

function GD(solution, scorer;
        learning_rate=.005,
        friction = 0.9,
        _steps_per_run = 500)
    """Nesterov Accelerated Gradient Descent is designed to prevent overshooting
    the local minima that is present in traditional gradient descent."""

    p = solution.line.coords
    momentum = zeros(length(p))
    for step in 1:_steps_per_run
        #apply the nesterov movement step
        gradient = learning_rate .* ForwardDiff.gradient(scorer, p)
        p += gradient

        #apply friction and the most recent velocity to momentum
        momentum *= friction
        momentum +=  gradient

        #make sure solution is always the best performing
        #coords found
        score = scorer(p)
        if score > solution.score
            new_line = CoordSystems.Line(p, solution.line.coord_system)
            solution = Solution(new_line, score)
        end
    end
    solution
end

function NAGD(solution, scorer;
        learning_rate=.005,
        friction = 0.9,
        _steps_per_run = 500)
    """Nesterov Accelerated Gradient Descent is designed to prevent overshooting
    the local minima that is present in traditional gradient descent."""

    p = solution.line.coords
    momentum = zeros(length(p))
    for step in 1:_steps_per_run
        #apply the nesterov movement step
        p += momentum
        gradient = learning_rate .* ForwardDiff.gradient(scorer, p)
        p += gradient

        #apply friction and the most recent velocity to momentum
        momentum *= friction
        momentum +=  gradient

        #make sure solution is always the best performing
        #coords found
        score = scorer(p)
        if score > solution.score
            new_line = CoordSystems.Line(p, solution.line.coord_system)
            solution = Solution(new_line, score)
        end
    end
    solution
end

function EvolutionaryStrategies(solution, scorer;
        n_children=500,
        range=.1,
        reduction_rate=1.0)
    """Evolutionary Strategy optimizer
    perturb each dimension in p by (standard gaussian * range)
    to generate a new child solution. Take the best child."""
    n_consecutive_fails = 1
    for _ in 1:n_children
        child = [d + (randn()*range) / (1 + n_consecutive_fails*reduction_rate)
                for d in solution.line.coords]
        score = scorer(child)
        if score > solution.score
            solution = Solution(
                CoordSystems.Line(child, solution.line.coord_system),
                score)
            n_consecutive_fails = 1
        else
            n_consecutive_fails += 1
        end
    end
    solution
end

evo_strat_params = PS.ParamSpaces(
    :n_children=> PS.Switch([
        PS.IntRange(1:10),
        PS.IntRange(10:100),
        PS.IntRange(100:1000),
        PS.IntRange(1000:10000),
        PS.IntRange(10000:100000),
    ]),
    :range=> PS.Switch([
        PS.Uniform(0, .1),
        PS.Uniform(0, .2),
        PS.Uniform(0, 1),
        PS.Uniform(0, 3)]),
    :reduction_rate=> PS.Switch([PS.Uniform(0, 1), PS.Uniform(0, 5)])
)

nagd_params = PS.ParamSpaces(
    :learning_rate=>PS.Switch([
        PS.Uniform(1e-5, 1e-4), PS.Uniform(1e-4, 1e-3),
        PS.Uniform(1e-3, 1e-2), PS.Uniform(1e-2, 1e-1),
        PS.Uniform(1e-1, 1),]),
    :friction=> PS.Switch([
        PS.Uniform(.99, 1), PS.Uniform(.9, 1), PS.Uniform(0, 1)])
)

optimizers = [
    Optimizer(EvolutionaryStrategies, evo_strat_params),
    Optimizer(NAGD, nagd_params),
    Optimizer(GD, nagd_params)
]

export Solution, optimizers

end
