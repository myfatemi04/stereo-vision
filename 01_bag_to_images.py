import pickle
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import tqdm
import numpy as np
import cv2
import json
import os

def extract_images(bag_dir, out_dir):
    camera_topics = []
    camera_conns = []
    total_msg_count = 0
    with Reader(bag_dir) as reader:
        for connection in reader.connections:
            if connection.msgtype == 'sensor_msgs/msg/CompressedImage':
                camera_topics.append(connection.topic)
                camera_conns.append(connection)
                total_msg_count += connection.msgcount
                
        print(f"Identified {len(camera_topics)} camera topics.")

        counters = {t: 0 for t in camera_topics}
        timestamps = {t: [] for t in camera_topics}

        # Extract subfolder names
        print("Output map:")
        outdirs = {}
        shortnames = {}
        for t in camera_topics:
            end = t.find("/image")
            start = t.rfind("/", 0, end) + 1
            shortnames[t] = t[start:end]
            outdirs[t] = os.path.join(out_dir, shortnames[t])
            
            print(t, "=>", outdirs[t])

            if not os.path.exists(outdirs[t]):
                os.makedirs(outdirs[t])
        
        with tqdm.tqdm(desc='Reading images.', total=total_msg_count) as pbar:
            for connection, timestamp, rawdata in reader.messages(camera_conns):
                # Check if the message topic matches.
                # msgtype should be sensor_msgs/msg/CompressedImage
                if (t := connection.topic) in camera_topics:
                    counters[t] += 1
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    np_arr = np.fromstring(msg.data, np.uint8)
                    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                    timestamps[t].append(timestamp)
                    path = f'{outdirs[t]}/image_{counters[t]}.png'
                    if not os.path.exists(path):
                        cv2.imwrite(path, image_np)

                    pbar.update(1)

        with open(f'{out_dir}/timestamps.json', 'w') as f:
            json.dump({shortnames[k]: v for k, v in timestamps.items()}, f)

def ros_lidar_to_numpy(parsed_message):
    points = np.frombuffer(parsed_message.data, np.float32).reshape((parsed_message.height, parsed_message.width, 6))
    point_xyzs = points[:, :, :3].reshape((-1, 3))
    point_intensities = points[:, :, 3].reshape((-1))
    point_rings = points[:, :, 4].reshape((-1))
    point_times = points[:, :, 5].reshape((-1))
    return (point_xyzs, point_intensities, point_rings, point_times)

def extract_lidar(bag_dir, out_dir):
    lidar_topics = []
    lidar_conns = []
    total_msg_count = 0
    with Reader(bag_dir) as reader:
        for connection in reader.connections:
            if connection.msgtype == 'sensor_msgs/msg/PointCloud2':
                lidar_topics.append(connection.topic)
                lidar_conns.append(connection)
                total_msg_count += connection.msgcount
                
        print(f"Identified {len(lidar_topics)} LiDAR topics.")

        counters = {t: 0 for t in lidar_topics}
        timestamps = {t: [] for t in lidar_topics}

        # Extract subfolder names
        print("Output map:")
        outdirs = {}
        shortnames = {}
        for t in lidar_topics:
            start = t.find("/", 2) + 1
            shortnames[t] = t[start:]
            outdirs[t] = os.path.join(out_dir, shortnames[t])
            
            print(t, "=>", outdirs[t])

            if not os.path.exists(outdirs[t]):
                os.makedirs(outdirs[t])
            
        # Iterate over messages
        with tqdm.tqdm(desc='Reading LiDAR point clouds', total=total_msg_count) as pbar:
            for connection, timestamp, rawdata in reader.messages(lidar_conns):
                # Check if the message topic matches
                # msgtype should be sensor_msgs/msg/PointCloud2
                if (t := connection.topic) in lidar_topics:
                    counters[t] += 1
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    data = ros_lidar_to_numpy(msg)
                    path = f'{outdirs[connection.topic]}/pcd_{counters[t]}.pkl'
                    if not os.path.exists(path):
                        with open(path, 'wb') as f:
                            pickle.dump(data, f)
                    pbar.update(1)
                    timestamps[t].append(timestamp)
        
        with open(f'{out_dir}/timestamps.json', 'w') as f:
            json.dump({shortnames[k]: v for k, v in timestamps.items()}, f)

'''
Topic: /vehicle_8/camera/front_left/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 16363 | Serialization Format: cdr
Topic: /vehicle_8/camera/front_left/image/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 3062 | Serialization Format: cdr
Topic: /vehicle_8/camera/front_left_center/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 16503 | Serialization Format: cdr
Topic: /vehicle_8/camera/front_left_center/image/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 5092 | Serialization Format: cdr
Topic: /vehicle_8/camera/front_right/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 16352 | Serialization Format: cdr
Topic: /vehicle_8/camera/front_right/image/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 4250 | Serialization Format: cdr
Topic: /vehicle_8/camera/rear_left/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 16324 | Serialization Format: cdr
Topic: /vehicle_8/camera/rear_left/image/compressed | Type: sensor_msgs/msg/CompressedImage | Count: 4803 | Serialization Format: cdr
Topic: /vehicle_8/camera/rear_right/camera_info | Type: sensor_msgs/msg/CameraInfo | Count: 16309 | Serialization Format: cdr
Topic: /vehicle_8/camera/rear_right/image/compressed
'''

extract_images('bags/ros/M-MULTI-SLOW-KAIST', 'bags/extracted/M-MULTI-SLOW-KAIST/camera')
extract_lidar('bags/ros/M-MULTI-SLOW-KAIST', 'bags/extracted/M-MULTI-SLOW-KAIST/lidar')
