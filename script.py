import rosbag, sensor_msgs.point_cloud2
import numpy as np
from matplotlib import pyplot as plt 
import raycasting_feature_extraction as rfe
import scan_preprocessors as pre
from copy import deepcopy

def take_x_and_y(points):
    #the velodyne point data is five dimensional? 
    #x y z ?? ??
    #we only want x and y so the other dimensions don't
    #interfere with math function, and so we need less memory
    #for storage
    return [[p[0], p[1]] for p in points]

def decimate(iterable, p=.1):
    new = []
    for i in iterable:
        if np.random.rand() < p:
            new.append(i)
    return new

def graph_line(line, plt):
    point_a_x = line.params["x"] + (np.cos(line.params["theta"]) * 10)
    point_a_y = line.params["y"] + (np.sin(line.params["theta"]) * 10)
    
    point_b_x = line.params["x"] + (np.cos(line.params["theta"]) * -10)
    point_b_y = line.params["y"] + (np.sin(line.params["theta"]) * -10)

    plt.plot([point_a_x, point_b_x], [point_a_y, point_b_y])

bag = rosbag.Bag('data_2020-06-10-10-24-18.bag')
for topic, msg, t in bag.read_messages(topics=["/velodyne_points"]):
    #get an appropriate amount of points
    points = list(sensor_msgs.point_cloud2.read_points(msg))
    points = decimate(points, .3)
    points = take_x_and_y(points)

    #graph points
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    # plt.scatter(x, y, s=.4)

    #run the minimum density selection
    dense_points = pre.min_density_selection(deepcopy(points), 5, .03)
    x = [p[0] for p in dense_points]
    y = [p[1] for p in dense_points]
    # plt.scatter(x, y, s=.4)

    #run the kmeans clustering selection
    centroids = pre.cluster_centroids(deepcopy(dense_points), 200)
    x = [p[0] for p in centroids]
    y = [p[1] for p in centroids]
    plt.scatter(x, y, s=1)

    #fit some lines 
    RFE = rfe.RFE(centroids, 100)
    RFE.remove_duplicate_lines()

    for rc in RFE.lines.population:
        graph_line(rc.line, plt)

    plt.show()
    input("")
bag.close()