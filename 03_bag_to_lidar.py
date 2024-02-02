import pickle
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import tqdm
import numpy as np
import json
import sys
import os

def ros_lidar_to_numpy(parsed_message):
    points = np.frombuffer(parsed_message.data, np.float32).reshape((parsed_message.height, parsed_message.width, 6))
    point_xyzs = points[:, :, :3].reshape((-1, 3))
    point_intensities = points[:, :, 3].reshape((-1))
    point_rings = points[:, :, 4].reshape((-1))
    point_times = points[:, :, 5].reshape((-1))
    return (point_xyzs, point_intensities, point_rings, point_times)

def extract(bag_dir, topic, output_dir):
    timestamps = []
    count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create reader instance and open for reading
    with Reader(bag_dir) as reader:
        # Iterate over messages
        with tqdm.tqdm(desc='Reading LiDAR points') as pbar:
            for connection, timestamp, rawdata in reader.messages():
                # Check if the message topic matches
                # msgtype should be sensor_msgs/msg/PointCloud2
                if connection.topic == topic:
                    count += 1
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    data = ros_lidar_to_numpy(msg)
                    path = f'{output_dir}/{count}.pkl'
                    if not os.path.exists(path):
                        with open(path, 'wb') as f:
                            pickle.dump(data, f)
                    pbar.update(1)
                    timestamps.append(timestamp)
    
    # Also store a `timestamps.json` file.
    with open(f'{output_dir}/timestamps.json', 'w') as f:
        json.dump(timestamps, f)

'''
Topic: /vehicle_8/luminar_front_points | Type: sensor_msgs/msg/PointCloud2 | Count: 11928 | Serialization Format: cdr
Topic: /vehicle_8/luminar_left_points | Type: sensor_msgs/msg/PointCloud2 | Count: 10635 | Serialization Format: cdr
Topic: /vehicle_8/luminar_right_points | Type: sensor_msgs/msg/PointCloud2 | Count: 9683 | Serialization Format: cdr
'''

def main():
    if len(sys.argv) != 4:
        print("Usage: python bag_to_images.py <bag_dir> <topic> <output_dir>")
        exit(1)

    bag_dir = sys.argv[1]
    topic = sys.argv[2]
    output_dir = sys.argv[3]
    extract(bag_dir, topic, output_dir)

if __name__ == '__main__':
    main()
