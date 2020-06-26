from matplotlib import pyplot as plt 

def graph_pointcloud(pointcloud):
    x = [p[0] for p in pointcloud.points]
    y = [p[1] for p in pointcloud.points]
    plt.scatter(x, y)

def show_graphs():
    plt.show()
    input("")

def graph_line_segment(pointA, pointB, points, color):
    x = [points[pointA][0], points[pointB][0]]
    y = [points[pointA][1], points[pointB][1]]
    plt.plot(x, y, c=color)


def graph_line_segments(polylines, pointcloud):
    colors = ["red", "orange", "blue", "purple"]
    color_index = 0 
    for polyline in polylines: 
        color_index = (color_index + 1) % (len(colors) - 1)
        for point in polyline:
            for otherpoint in polyline[point]:
                graph_line_segment(point, otherpoint, pointcloud.points, colors[color_index])

def graph_polylines(polylines, pointcloud):
    colors = ["red", "orange", "blue", "purple"]
    color_index = 0 
    for polyline in polylines: 
        color_index = (color_index + 1) % (len(colors) - 1)
        prev_point = polyline[0]
        for i in range(1, len(polyline) - 1):
            point = polyline[i]
            graph_line_segment(prev_point, point, pointcloud.points, colors[color_index])
            prev_point = point

def graph_slope_intercept_lines(lines, pointcloud):
    colors = ["red", "orange", "blue", "purple"]
    color_index = 0 
    for line in lines:
        color_index = (color_index + 1) % (len(colors) - 1)
        x0 = -5
        x1 = 5
        y0 = line[0]*x0 + line[1]
        y1 = line[0]*x1 + line[1]
        plt.plot([x0, x1], [y0, y1], c=colors[color_index])