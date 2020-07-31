module LS

include("coord_systems.jl")
include("gain_functions.jl")
include("optimizers.jl")
include("hyperparam_spaces.jl")

mutable struct LocalSearch
    scorer #a gain func associated with a point cloud
    optimizer
    solution::Optimizers.Solution
end

function LocalSearch(
        gain_func::GainFunctions.GainFunction,
        points::Matrix{Float64},
        optimizer::Optimizers.Optimizer,
        init_line::Optimizers.CoordSystems.Line)
    """Construct a scorer closure and use it to make a LocalSearch
    instance"""
    scorer(params) = GainFunctions.score(
                        init_line.coord_system.from(params)...,
                        points,
                        gain_func)
    LocalSearch(
        scorer,
        optimizer,
        Optimizers.Solution(init_line, scorer(init_line.coords))
    )
end

function give_runtime(ls::LocalSearch, tl)
    """run the passed local search for the passed amount of time"""
    start_time = time()
    while time() - start_time < tl
        ls.solution = ls.optimizer.optimizer(ls.solution, ls.scorer)
    end
    ls
end

end
