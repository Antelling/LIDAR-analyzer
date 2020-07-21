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

 function ES(p, scorer; range=.1, n_children=5, max_fails=5)
     """Evolutionary Strategy optimizer
     perturb each dimension in p by (standard gaussian * range)
     to generate a new child solution. Take the best child then repeat."""
     fails = 0
     best_found_solution = [p, scorer(p)]
     while fails <= max_fails
         best_found_child = [[], 0]
         for _ in 1:n_children
             new_child = [d + randn() * range/(fails+1) for d in p]
             new_score = scorer(new_child)
             if new_score > best_found_child[2]
                 best_found_child = [new_child, new_score]
             end
         end
         if best_found_child[2] > best_found_solution[2]
             best_found_solution = best_found_child
             fails = 0
         else
             fails += 1
         end
     end
     best_found_solution[1]
 end

function graph_point_ray(p, points, score)
    x1, y1, theta, len = p
    x2 = x1 + cos(theta)*len
    y2 = y1 + sin(theta)*len

    m = (y2 - y1)/(x2 - x1)
    b = y1 - m*x1
    l = min(x1, x2)
    h = max(x1, x2)

    x = points[:, 1]
    y = points[:, 2]
    scatter!(x, y, title = "Score: $(score)")


    plot!([l, h], [m*l+b, m*h+b])
end

function opt(p, scorer)
    optimized = NAG(copy(p), scorer,
        learning_rate=.00001,
        max_steps=99999)
    optimized = NAG(copy(p), scorer,
        learning_rate=.000001,
        max_steps=99999)
    optimized = NAG(copy(p), scorer,
        learning_rate=.0000001,
        max_steps=99999)
    optimized
end

function slope_intercept_bounds_scorer(p)
    """interpret the four parameters as m, b, min, max"""
    m, b, l, h = p
    total_score(m, b, l, h, points,
        smoother=1,
        score_radius=5,
        length_scale=.1,
        length_exponent=.5,
        distance_exponent=1.2,
        sum_exponent=2)
end

function point_ray_scorer(p)
    """interpret the four parameters as x, y, theta, length"""
    x1, y1, theta, len = p
    x2 = x1 + cos(theta)*len
    y2 = y1 + sin(theta)*len

    m = (y2 - y1)/(x2 - x1)
    b = y1 - m*x1
    l = min(x1, x2)
    h = max(x1, x2)

    total_score(m, b, l, h, points,
        smoother=1,
        score_radius=5,
        length_scale=.1,
        length_exponent=.5,
        distance_exponent=1.2,
        sum_exponent=2)
end

function two_point_scorer(p)
    """interpret the four parameters as x1, y1, x2, y2"""
    x1, y1, x2, y2 = p
    m = (y2 - y1)/(x2 - x1)
    b = y1 - m*x1
    l = x1
    h = x2
    total_score(m, b, l, h, points,
        smoother=1,
        score_radius=5,
        length_scale=.1,
        length_exponent=.5,
        distance_exponent=1.2,
        sum_exponent=2)
end

include("./point_loader.jl")
points = point_loader.single_line()
point_loader.graph(points)

initial = [5, -.2, 20, 5]
graph_point_ray(initial, points, point_ray_scorer(initial))

optimized = ES(copy(initial), point_ray_scorer, range=.1, n_children=50000, max_fails=30)
graph(optimized, points, scorer(optimized))

hand_opt = [5.9, -.38, 15.2, -.195]
graph(hand_opt, points, scorer(hand_opt))
