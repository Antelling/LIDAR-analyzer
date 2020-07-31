using Plots, JSON
include("coord_systems.jl")
include("mock_data.jl")
include("gain_functions.jl")
include("optimizers.jl")
include("hyperparam_spaces.jl")
include("local_search.jl")

struct Configuration
    gf::LS.GainFunctions.GainFunction
    oz::LS.Optimizers.Optimizer
    score::Union{Float64,Nothing}
end
function Configuration(config::Configuration, score::Float64)
    Configuration(config.gf, config.oz, score)
end
function Configuration()
    """generate a Configuration struct from the export lists of the
    GainFunction and Optimizer modules"""
    optimizer = rand(LS.Optimizers.optimizers)
    optimizer = LS.Optimizers.resample(optimizer)
    gain_function = rand(LS.GainFunctions.gain_functions)
    gain_function = LS.GainFunctions.resample(gain_function)
    Configuration(gain_function, optimizer, nothing)
end

#generate and graph some mock data
md = MockData.corner()
MockData.graph_mock_data(md, title="Simulated Results of \n
    KMeans and Undersampling Preprocessing")

#utilities to display configurations
function round_dict(dict::Dict; digits=2)
    return [key=>round(value, digits=digits) for (key, value) in dict]
end
function truncate_show(config::Configuration)
    print("Configuration(")
    truncate_show(config.gf)
    truncate_show(config.oz)
    println(config.score, ")")
end
function truncate_show(gf::LS.GainFunctions.GainFunction)
    print("GainFunction(", gf.formula, ", ", round_dict(gf.params), ") ")
end
function truncate_show(opt::LS.Optimizers.Optimizer)
    print("Optimizer(", opt.optimizer, ", ", round_dict(opt.params), ") ")
end


function generate_starting_set(n_starts::Integer, data::MockData.MockedData)
    """make the set of starting positions that will be used for comparisons"""
    truth_error_pairs_sets = []
    for n in 1:n_starts
        truth_error_pairs = []
        for line in data.labels
            il = copy(line.coords) .+ 3*(rand(length(line.coords)) .- .5)
            error_line = LS.Optimizers.CoordSystems.Line(
                    il, CoordSystems.two_points)
            push!(truth_error_pairs, [line, error_line])
        end
        push!(truth_error_pairs_sets, truth_error_pairs)
    end
    truth_error_pairs_sets
end

function randomly_try_params(
        data::MockData.MockedData,
        gain_functions::Vector{GainFunctions.GainFunction},
        optimizers::Vector{Optimizers.Optimizer};
        n_trials=50, n_starts_per_trial=10, optimizer_runtime=.5)
    """for n in n_samples: Randomly choose a gain_function and optimizer,
    randomly sample params for them, then score them. Return the vector of
    (score, params) results.
    Params:
        data -> MockedData to test on
        gain_functions -> vector of possible gain_functions to sample from
        optimizers -> vector of possible optimizers to sample from
        n_samples -> amount of trials to run
        n_starts_per_trial -> amount of times wrong mockeddata labels should
                be generated
        optimizer_runtime -> how much time to give each optimizer"""

    #get the starting positions
    truth_error_pairs_sets = generate_starting_set(n_starts_per_trial, data)

    results::Vector{Configuration} = [] #all results will be saved

    #run the trials
    println("starting trials...")
    lowest_found_error = 9999999999
    Threads.@threads for n in 1:n_trials

        config = Configuration() #generate a new configuration

        #sum up the total error and elapsed time for every per-trial-start
        configuration_total_error = 0
        total_elapsed_time = 0
        for truth_error_pairs in truth_error_pairs_sets
            for (true_line, error_line) in truth_error_pairs

                #create the local search object then run the search
                ls = LS.LocalSearch(
                        config.gf, data.points, config.oz, error_line)
                st = time()
                LS.give_runtime(ls, optimizer_runtime)
                total_elapsed_time += time() - st

                #get distance between the discovered line and the truth
                optimized_line = ls.solution.line
                standard_coord_form = optimized_line.coord_system.from(
                    optimized_line.coords)
                two_point_opt_form = CoordSystems.two_points.to(
                    standard_coord_form)
                two_point_truth_form = CoordSystems.two_points.to(
                    true_line.coord_system.from(true_line.coords))
                dist = hypot((two_point_truth_form .- two_point_opt_form)...)

                #add this to the total
                configuration_total_error += dist
            end
        end

        #some optimizers take a long amount of time with some parameters.
        #we have a crude time limit check, so we should also weight the
        #scores according to elapsed time
        configuration_total_error *= total_elapsed_time
        #that number will be hard to interpret, so lets scale
        #it back down into its original magnitude
        lower_bound_elapsed_time = n_starts_per_trial * optimizer_runtime
        configuration_total_error /= lower_bound_elapsed_time
        #and, lets take the average
        configuration_total_error /= (n_starts_per_trial *
                length(truth_error_pairs_sets[1]))

        #save this result
        scored_config = Configuration(config, configuration_total_error)
        push!(results, scored_config)

        #check if this is the best so far, for the real time display
        if configuration_total_error < lowest_found_error
            truncate_show(scored_config)
            lowest_found_error = configuration_total_error
        end
    end
    sort!(results, by=x->x.score)
    results
end

function construct_serialized_dict(results::Vector{Configuration})
    construct_serialized_dict.(results)
end

function construct_serialized_dict(config::Configuration)
    serializer = Dict()
    for key in [:gf, :oz, :score]
        val = getproperty(config, key)
        serializer[key] = construct_serialized_dict(val)
    end
    serializer
end

function construct_serialized_dict(num::Number)
    num
end

function construct_serialized_dict(d::Dict)
    d
end

function construct_serialized_dict(opt::LS.Optimizers.Optimizer)
    serializer = Dict()
    serializer[Symbol("opt_name")] = "$(opt.optimizer)"
    for (key, val) in opt.params
        prefixed_key = Symbol("opt_$(key)")
        serializer[prefixed_key] = construct_serialized_dict(val)
    end
    serializer
end


function construct_serialized_dict(gf::LS.GainFunctions.GainFunction)
    serializer = Dict()
    serializer[Symbol("gf_name")] = "$(gf.formula)"
    for (key, val) in gf.params
        prefixed_key = Symbol("gf_$(key)")
        serializer[prefixed_key] = construct_serialized_dict(val)
    end
    serializer
end

function construct_serialized_dict(n::Nothing)
    nothing
end

function pp(config::Configuration)
    "Configuration(GainFunction($(config.gf.formula), " *
        "$(JSON.json(config.gf.params))), \n\t" *
    "Optimizer($(config.oz.optimizer), $(JSON.json(config.oz.params)))), \n\t" *
    "Score($(config.score))"
end

function pp(results::Vector{Configuration})
    join(serialize.(results), "\n")
end


truth_error_pairs_sets = generate_starting_set(5, md)
true_line, error_line = truth_error_pairs_sets[1][1]

config = Configuration(
    LS.GainFunctions.gain_functions[1],
    LS.Optimizers.optimizers[2],
    nothing)
ls = LS.LocalSearch(config.gf, md.points, config.oz, error_line)
LS.give_runtime(ls, 5)
construct_serialized_dict(config)

results = randomly_try_params(md, GainFunctions.gain_functions,
        Optimizers.optimizers,
        n_starts_per_trial=4,
        n_trials=5,
        optimizer_runtime=1)

JSON.json(construct_serialized_dict(results))

get_after_it = true
if get_after_it
    results = randomly_try_params(md, GainFunctions.gain_functions,
        Optimizers.optimizers,
        n_starts_per_trial=1,
        n_trials=1,
        optimizer_runtime=.01)

    results = randomly_try_params(md, GainFunctions.gain_functions,
        Optimizers.optimizers,
        n_starts_per_trial=4,
        n_trials=2500,
        optimizer_runtime=1)

    file = open("res.json", "w")
    write(file, JSON.json(construct_serialized_dict(results), 4))
    close(file)
end

println("File Ran Successfully")
