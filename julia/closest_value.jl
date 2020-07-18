using ForwardDiff, Plots

function value_closest_to_point(m, b, x, y)
    """Find the x value where the line described by m and
    b passes closest to the (x, y) point """
    (-b * m + m * y + x) / (m^2 + 1)
end


function score(m, b, x, y, l, h; s=.01, r=1, ex=2)
    v = value_closest_to_point(m, b, x, y)
    v = min(h, v)
    v = max(l, v)
    d = s + hypot(v - x, m*v + b - y)
    if d^ex < r
        return r/d^ex
    else
        return 0
    end
end

function total_score(m, b, l, h, points;
        smoother=.01,
        length_scale=.2,
        score_radius=2,
        distance_exponent=2,
        length_exponent=.5,
        sum_exponent=2)
     len = hypot(l - h, (m*l + b) - (m*h + b)) * length_scale
     vs = [score(m, b, p[1], p[2], l, h, s=smoother, r=score_radius, ex=distance_exponent) for p in eachrow(points)]
     s = sum(vs)
     return (s^sum_exponent)/(len^length_exponent)
 end

function NAG(
     p,
     scorer;
     learning_rate = 0.05,
     allowable_failures::Int = 5,
     max_steps=99999,
     friction = 0.9,
 )
     best_found = [copy(p), scorer(p)]
     prev_score = -1
     current_score = 0
     momentum = zeros(length(p))
     failures = 0
     steps = 0
     while failures <= allowable_failures && steps < max_steps
         steps += 1

         p += momentum
         gradient = learning_rate .* ForwardDiff.gradient(scorer, p)
         p += gradient

         momentum *= friction
         momentum +=  gradient

         prev_score = current_score
         current_score = scorer(p)
         if current_score > best_found[2]
             best_found = [copy(p), current_score]
             failures = 0
         else
             failures += 1
         end
     end
     best_found[1]
 end

function graph(params, points)
    m, b, l, h = params
    score = total_score(m, b, l, h, points)

    x = points[:, 1]
    y = points[:, 2]
    scatter(x, y, title = "Score: $(score)")


    plot!([l, h], [m*l+b, m*h+b])
end

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

total_score(1, 0, 0, 5.7, rpoints)

function opt(p, scorer)
    optimized = NAG(copy(p), scorer,
        learning_rate=.000001,
        max_steps=99999)
    optimized = NAG(optimized, scorer,
        learning_rate=.0001,
        max_steps=99999)
    optimized
end

scorer(p) = total_score(p[1], p[2], p[3], p[4], rpoints,
    smoother=.1,
    score_radius=5,
    length_scale=.02,
    length_exponent=1,
    distance_exponent=1.2,
    sum_exponent=1)

initial1 = [.02, 1.9, 8, 15]
initial2 = [.02, -1.5, 6, 11]
initial3 = [-.02, -1.5, 1, 8]
graph(initial3, rpoints)
optimized = opt(initial3, scorer)
graph(optimized, rpoints)
elongated_optimized = copy(optimized)
elongated_optimized[4] += 1
graph(elongated_optimized, rpoints)
elongated_optimized = opt(elongated_optimized, scorer)
