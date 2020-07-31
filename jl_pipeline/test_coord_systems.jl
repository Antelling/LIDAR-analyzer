include("coord_systems.jl")

l1 = [1, 0, 0, 1]
line1 = CoordSystems.Line(l1, CoordSystems.slope_intercept_bounds)
@assert line1.coord_system.from(line1.coords) == l1

l2 = [0, 0, 1, 1]
line2 = CoordSystems.Line(l2, CoordSystems.two_points)
line2.coord_system.from(line2.coords)
@assert line2.coord_system.from(line2.coords) == l1
@assert line2.coord_system.to(l1) == line2.coords


mblh = [.5, 1, 5, 10]
xyxy = [5, 3.5, 10, 6]

@assert CoordSystems.two_points.from(xyxy) == mblh
@assert CoordSystems.two_points.to(mblh) == xyxy
