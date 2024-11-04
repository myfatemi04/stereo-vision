import argparse
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String
from sensor_msgs.msg import Image
import rosbag2_py
import PIL.Image
import os
import numpy as np

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

unique_types = set()
unique_sizes = set()

for topic, msg, timestamp in read_messages(input_file):
    tn = type(msg).__name__
    if tn == 'Image':
        # print(msg.width, msg.height)
        # Save this image.
        width = msg.width
        height = msg.height
        array = np.array(msg.data).reshape((height, width, 3))
        PIL.Image.fromarray(array).save("first_image.png")
        print("Image size:", (height, width))
        break
        
    if tn not in unique_types:
        print(tn, msg)
        unique_types.add(tn)
        
    # if isinstance(msg, String):
    #     print(f"{topic} [{timestamp}]: '{msg.data}'")
    # else:
    #     print(f"{topic} [{timestamp}]: ({type(msg).__name__})")
