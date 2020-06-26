import rosbag, sensor_msgs.point_cloud2

class DataLoader(object):
    def __init__(self, filename):
        self.bag = rosbag.Bag(filename)

    def load_next_frame(self):
        topic, msg, time = next(self.bag.read_messages(topics=["/velodyne_points"]))
        return list(sensor_msgs.point_cloud2.read_points(msg))