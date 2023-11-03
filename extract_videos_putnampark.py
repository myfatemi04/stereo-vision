import argparse
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String
from sensor_msgs.msg import Image
import rosbag2_py
import PIL.Image
import os
import numpy as np
import cv2
import json

def read_messages(input_bag: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_bag, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    topic_types = reader.get_all_topics_and_types()

    def typename(topic_name):
        for topic_type in topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        msg_type = get_message(typename(topic))
        msg = deserialize_message(data, msg_type)
        yield topic, msg, timestamp
    del reader

(fname,) = [f for f in os.listdir("run4-2-camera/") if '.mcap' in f]
input_file = f"run4-2-camera/{fname}"

count = 0
timestamps = {}
video_writers = {}

# Iterate over messages
for topic, msg, timestamp in read_messages(input_file):
    tn = type(msg).__name__
    if tn == 'Image':
        width, height = 516, 384
        camera_id = topic.split("/")[2]
        
        count += 1
        print("Frames processed:", count, end='\r')
        if camera_id not in timestamps:
            print("Adding", camera_id)
            # Define the codec and create a VideoWriter instance for the camera
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs if needed
            # out_file = f'{camera_id}.mp4'
            # video_writers[camera_id] = cv2.VideoWriter(out_file, fourcc, 20, (width, height))
            timestamps[camera_id] = []
        
        # array = np.array(msg.data).reshape((height, width, 3))
        # video_writers[camera_id].write(array)

        # Write the image frame to the corresponding video file
        timestamps[camera_id].append(timestamp)

# Release all VideoWriter instances
# for camera_id, writer in video_writers.items():
#     writer.release()

with open("timestamps.json", "w") as f:
    json.dump(timestamps, f)
