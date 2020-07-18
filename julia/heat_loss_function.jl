using Plots, TaylorSeries

function distance_to_heat(distance, s::Number=.01)
    """a point source acts according to the inverse square law"""
    return 1/(s + distance^2)
end

function value_to_distance(value, m, b, x, y)
    """calculate the y value of a line given an x value, then return the
    distance of that point to the passed (x, y) point"""
    x_error = value - x
    y_error = (m*value + b) - y
    return sqrt(x_error^2 + y_error^2)
end

function value_to_heat(value, m, b, x, y)
    """return how hot a line is at a certain value"""
    return distance_to_heat(value_to_distance(value, m, b, x, y))
end

function arclength_value_to_heat(value, m, b, x, y)
    """combine the heat calculation with the arclength formula"""
    """arclength formula is integral of hypot(y', x'). Normally we set
    y' to 1 then x' is simply dy/dx. Since we are dealing with a line,
    dy/dx is always m. In order to combine the arclength formula with the
    heat value, we simply multiply heat by each derivative term. Integration
    will be handled by a taylor series, so this function just needs to
    represent the inner term. """
    heat = value_to_heat(value, m, b, x, y)
    return sqrt(heat^2 + (m*heat)^2)
end

function unrolled_heat(v, m, b, x, y; s=.01)
    sqrt(
        (
            1/(s +
                sqrt((v - x)^2 + ((m*v + b) - y)^2)
            ^2)
        )^2 +
        (
            m/(s +
                sqrt((v - x)^2 + ((m*v + b) - y)^2)
            ^2)
        )^2
    )
end

function value_closest_to_point(m, b, x, y)
    """Find the x value where the line described by m and
    b passes closest to the (x, y) point """
    (-b * m + m * y + x) / (m^2 + 1)
end

function find_heat_tails(m, b, x, y, l, h; epsilon=.01, precision=.01)
    """The heat function is shaped like a big spike with
    a rounded top. The tails approach 0. This function finds
    the largest possible interval where all values included
    are greater than epsilon, the low value is > l, and the
    high value is < h."""
    #this is definitely able to be solved analytically,
    #but it would require dhardcoding the solution to the
    #heat function
    #I should probably switch to representing functions with some
    #sort of symbolic math library? Or if this works well hardcoding
    #is fine I guess. For now we can do a half-interval search in one
    #direction, and then we know the function is symmetric...
    center_value = value_closest_to_point(m, b, x, y)
    step = .001
    while arclength_value_to_heat(center_value + step, m, b, x, y) > epsilon
        step *= 2
    end
    low_val = step/2
    high_val = step
    while (high_val - low_val) > precision
        middle_val = low_val + (high_val - low_val)/2
        if arclength_value_to_heat(center_value + middle_val, m, b, x, y) > epsilon
            #if the value is too high, we need to go further out
            low_val = middle_val
        else
            high_val = middle_val
        end
    end
    low = max(l, center_value - high_val)
    high = min(h, center_value + high_val)
    return [low, high]
end

function scale_function(func, l, h)
    """return a new function that is similar
    to func.[l, h] over [-1, 1]"""
    range = h - l
    center = l + range/2
    scaled_func(v) = func(v*(range/2) + center)
    scaled_func
end

function unrolled_heat_with_scaling(v, m, b, x, y, range, center; s=.01)
    """The taylor series library seems to not be able to handle
    the functions returned by the scale_function method. This combines the
    math done by that method with the heat calculation. """
    v = v*(range/2) + center
    sqrt(
        (
            1/(s +
                sqrt((v - x)^2 + ((m*v + b) - y)^2)
            ^2)
        )^2 +
        (
            m/(s +
                sqrt((v - x)^2 + ((m*v + b) - y)^2)
            ^2)
        )^2
    )
end

f(x) = max(1/(abs(x+5) + 1), 1/(abs(x-5) + 1))
scaled = scale_function(f, -10, 10)

function total_heat(m, b, x, y, l, h;
        order=64,
        s=.01
        )
    """return the total heat exerted on the line from the values
    l to h"""
    """There is no elementary solution for the integral of the
    arclength_value_to_heat function, so we need to approximate it.
    We can't approximate it numerically, because this function needs
    to work with autodiff methods. We will use a taylor series."""
    #get the range of values we care about
    l, h = find_heat_tails(m, b, x, y, l, h)
    range = h - l
    center = l + range/2

    #create a closure with the values we want
    heat(v) = unrolled_heat_with_scaling(v, m, b, x, y, range, center, s=s)

    #create the approximation
    t = Taylor1(typeof(l), order)
    ts_polynomial = heat(t)

    #return a function that evaluates the polynomial
    ts_approximation(v) = evaluate(ts_polynomial, v)
    ts_approximation, heat
end

m, b, x, y, l, h = 1.0, 0.0, 1.0, 2.0, -15.0, 15.0
unrolled_heat(1.6, m, b, x, y)
appr, scaled = total_heat(m, b, x, y, l, h, order=200)
appr(1.2)
truth(v) = arclength_value_to_heat(v, m, b, x, y)
truth(1.6)

plot(truth, -2, 5)
plot!(scaled, -1, 1)
plot!(appr, -.5, .5)

function approx_and_graph(f)
    t = Taylor1(typeof(l), 5)
    ts_polynomial = f(t)
    ts(v) = evaluate(ts_polynomial, v)
    plot!(ts, -.4, .4)
end

scaled(x) = scale_function(truth, -10, 10)
plot(scaled, -.4, .4)
approx_and_graph(scaled)
println("loaded")
