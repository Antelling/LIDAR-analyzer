module MockData
include("coord_systems.jl")

struct MockedData
    labels::Vector{CoordSystems.Line}
    points::Matrix{Float64}
end

function add_line(points, frequency, first_point, second_point, std)
    """add points sampled with noise from the line represented
    by the two points to the list of points, then return the (m b l h)
    line parameter tuple. """
    length = hypot((first_point - second_point)...)
    n = round(frequency * length)
    range = abs(first_point[1] - second_point[1])
    m, b, l, h = CoordSystems.two_points.from(first_point..., second_point...)

    for p in 1:n
        x = l + (p/n)*range
        y = m*x + b + std * (rand() - .5)
        push!(points, [x, y])
    end
    CoordSystems.Line([first_point..., second_point...], CoordSystems.two_points)
end


function corner()
    gen_point() = [rand(1:100), rand(1:100)]
    start, vertex, endp = gen_point(), gen_point(), gen_point()

    points = Vector{Vector{Float64}}()
    labels = []
    push!(labels, add_line(points, .5, start, vertex, .4))
    push!(labels, add_line(points, .8, vertex, endp, .4))
    pointcloud = rotl90(hcat(points...))
    return MockedData(labels, pointcloud)
end

using Plots
function graph_mock_data(md::MockedData; title="Mocked Data")
    x = md.points[:, 1]
    y = md.points[:, 2]
    scatter(x, y)
    title!(title)

    # lines = [line.coords for line in md.labels]
    # for (x1, y1, x2, y2) in lines
    #     plot!([x1, x2], [y1, y2])
    # end
end

export MockedData, corner, graph_mock_data
end
