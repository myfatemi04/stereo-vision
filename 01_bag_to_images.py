from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import tqdm
import numpy as np
import cv2
import json
import sys
import os

def extract(bag_dir, topic, output_dir):
    timestamps = []
    count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create reader instance and open for reading
    with Reader(bag_dir) as reader:
        # Iterate over messages
        with tqdm.tqdm(desc='Reading images.') as pbar:
            for connection, timestamp, rawdata in reader.messages():
                # Check if the message topic matches.
                # msgtype should be sensor_msgs/msg/CompressedImage
                if connection.topic == topic:
                    count += 1
                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    np_arr = np.fromstring(msg.data, np.uint8)
                    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                    timestamps.append(timestamp)
                    path = f'{output_dir}/{count}.png'
                    if not os.path.exists(path):
                        cv2.imwrite(path, image_np)
                    pbar.update(1)
    
    # Also store a `timestamps.json` file.
    with open(f'{output_dir}/timestamps.json', 'w') as f:
        json.dump(timestamps, f)

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

# python 01_bag_to_images.py M-MULTI-SLOW-KAIST /vehicle_8/camera/front_left/image/compressed M-MULTI-SLOW-KAIST_images/front_left
# python 01_bag_to_images.py M-MULTI-SLOW-KAIST /vehicle_8/camera/front_right/image/compressed M-MULTI-SLOW-KAIST_images/front_right

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
