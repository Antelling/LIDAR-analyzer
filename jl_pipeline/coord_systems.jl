module CoordSystems
"""Provide a set of conversions that accept lines in their
described coordinate systems and returns the same line represented
in the slope_intercept_bounds coordinate system"""

struct CoordSystem
    """A set of functions providing conversions between the standard
    coordinate system and this one. """
    # a function to convert from the special space to the standard space
    from::Function
    # a function to convert from standard to this
    to::Function
end

struct Line
    """Associate a set of coordinates with a coordinate system"""
    coords::Vector{<:Number}
    coord_system
end

function not_implemented(p)
    error("not implemented")
end

function from_slope_intercept_bounds(p)
    """interpret the four parameters as m, b, min, max

    m: slope of the line
    b: y-intercept of the line
    min: lower endpoint x-value of the line
    max: upper endpoint x-value of the line """
    m, b, l, h = p
    low = min(l, h)
    high = max(l, h)
    return [m, b, low, high]
end

function from_point_ray(p)
    """interpret the four parameters as x, y, theta, length

    x: x-value of start point
    y: y-value of start point
    theta: value in radians of slope of the ray
    length: how long the ray continues out from (x, y)"""
    x1, y1, theta, len = p
    x2 = x1 + cos(theta)*len
    y2 = y1 + sin(theta)*len

    m = (y2 - y1)/(x2 - x1)
    b = y1 - m*x1
    l = min(x1, x2)
    h = max(x1, x2)

    return [m, b, l, h]
end
function from_two_points(p)
    from_two_points(p...)
end
function from_two_points(x1, y1, x2, y2)
    """interpret the four parameters as x1, y1, x2, y2

    (x1, y1) is the start/end point of the line
    (x2, y2) is the end/start point of the line"""
    m = (y2 - y1)/(x2 - x1)
    b = y1 - m*x1
    l = min(x1, x2)
    h = max(x1, x2)

    return [m, b, l, h]
end

to_two_points(p) = to_two_points(p...)
function to_two_points(m, b, l, h)
    low_y = m * l + b
    high_y = m * h + b
    return [l, low_y, h, high_y]
end

using Plots
function graph(line::Line)
    standard_coords = line.coord_system.from(line.coords)
    two_points = to_two_points(standard_coords...)
    plot!([two_points[1], two_points[3]], [two_points[2], two_points[4]])
end

slope_intercept_bounds = CoordSystem(from_slope_intercept_bounds, not_implemented)
point_ray = CoordSystem(from_point_ray, not_implemented)
two_points = CoordSystem(from_two_points, to_two_points)

export CoordSystem, Line, slope_intercept_bounds, point_ray, two_points, graph
end
