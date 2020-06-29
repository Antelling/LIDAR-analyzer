from matplotlib import pyplot as plt 

def graph_pointcloud(pointcloud, s=1, c="blue", title="points"):
    x = [p[0] for p in pointcloud.points]
    y = [p[1] for p in pointcloud.points]
    plt.scatter(x, y, s=s, c=c, label=title)

def show_graphs(title=""):
    plt.legend()
    plt.title(title)
    plt.show()

def graph_line_segment(pointA, pointB, points, color):
    x = [points[pointA][0], points[pointB][0]]
    y = [points[pointA][1], points[pointB][1]]
    plt.plot(x, y, c=color)


def graph_line_segments(linesegments, pointcloud, colors=None):
    if colors is None:
        colors = ["red", "orange", "blue", "purple"]
    color_index = 0 
    for key in linesegments: 
        color_index = (color_index + 1) % (len(colors))
        for otherline in linesegments[key]:
            graph_line_segment(key, otherline, pointcloud.points, colors[color_index])

def graph_polylines(polylines, pointcloud, colors=None):
    if colors is None:
        colors = ["red", "orange", "blue", "purple"]
    color_index = 0 
    for polyline in polylines: 
        color_index = (color_index + 1) % (len(colors))
        prev_point = polyline[0]
        for i in range(1, len(polyline) - 1):
            point = polyline[i]
            graph_line_segment(prev_point, point, pointcloud.points, colors[color_index])
            prev_point = point

def graph_slope_intercept_lines(lines, pointcloud, colors=None, l=1):
    if colors is None:
        colors = ["red", "orange", "blue", "purple"]
    color_index = 0 
    for line in lines:
        color_index = (color_index + 1) % (len(colors))
        x0 = line.params["low"]
        x1 = line.params["high"]
        y0 = line.params["m"]*x0 + line.params["b"]
        y1 = line.params["m"]*x1 + line.params["b"]
        plt.plot([x0, x1], [y0, y1], c=colors[color_index], linewidth=l)