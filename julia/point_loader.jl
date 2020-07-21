module point_loader
export complete_test_frame, single_line, graph

using PyCall
push!(pyimport("sys")."path", pwd());

pointcloud = pyimport("pointcloud")
dataloader = pyimport("data_handler")

function complete_test_frame()
    dl = dataloader.DataLoader("data_2020-06-10-10-24-18.bag")
    pc = pointcloud.Pointcloud(dl.load_next_frame())
    pc.remove_floor(floor = 0.05)
    pc.take_xy()
    pc.take_percentage(0.5)
    pc.biased_undersample(percentile = 0.1, radius = 0.6)
    pc.take_centroids(400, exact = true)
    pc.points
end

function single_line(m=.02, b=-.5, x_vals=6:.2:15, noise=-.01)
    points = [[x, m*x+b + (rand()-.5)*noise] for x in x_vals]
    rotl90(hcat(points...))
end

using Plots
function graph(points)
    x = points[:, 1]
    y = points[:, 2]
    scatter(x, y)
end
# graph(complete_test_frame())
# graph(single_line())
end
