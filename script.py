import rosbag, sensor_msgs.point_cloud2
import numpy as np
from matplotlib import pyplot as plt 
import line_fit

def graph_line(m, b):
    x = np.linspace(-5,25,100)
    y = x*m+b
    plt.plot(x, y)

def decimate(iterable):
    new = []
    for i in iterable:
        if np.random.rand() < .1:
            new.append(i)
    return new

bag = rosbag.Bag('data_2020-06-10-10-24-18.bag')
for topic, msg, t in bag.read_messages(topics=["/velodyne_points"]):
    points = decimate(list(sensor_msgs.point_cloud2.read_points(msg)))
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.scatter(x, y, s=.4)

    lines = line_fit.top_n_lines(points, 12)
    for line in lines:
        graph_line(line[0], line[1])
    plt.show()
    input("")
bag.close()