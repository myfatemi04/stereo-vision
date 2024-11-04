import json
import os
import pickle
from collections import defaultdict

import click
import cv2
import numpy as np
import tqdm
from rosbags.interfaces import Connection
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr


def identify_topics(reader: Reader):
    topics_by_type: dict[str, list[Connection]] = {}
    for connection in reader.connections:
        if connection.msgtype not in topics_by_type:
            topics_by_type[connection.msgtype] = []
        topics_by_type[connection.msgtype].append(connection)
    return topics_by_type


def infer_camera_name(camera_topic_name: str):
    end = camera_topic_name.find("/image")
    start = camera_topic_name.rfind("/", 0, end) + 1
    return camera_topic_name[start:end]


def infer_lidar_name(lidar_topic_name: str):
    start = lidar_topic_name.find("/", 2) + 1
    return lidar_topic_name[start:]


def compressed_image_message_to_numpy(rawdata: bytes, connection: Connection):
    msg = deserialize_cdr(rawdata, connection.msgtype)
    data: bytes = msg.data  # type: ignore
    np_arr = np.fromstring(data, dtype=np.uint8)  # type: ignore
    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    return image_np


def ros_lidar_to_numpy(parsed_message):
    points = np.frombuffer(parsed_message.data, np.float32).reshape(
        (parsed_message.height, parsed_message.width, 6)
    )
    point_xyzs = points[:, :, :3].reshape((-1, 3))
    point_intensities = points[:, :, 3].reshape((-1))
    point_rings = points[:, :, 4].reshape((-1))
    point_times = points[:, :, 5].reshape((-1))
    return (point_xyzs, point_intensities, point_rings, point_times)


@click.command()
@click.option("--bag_dir", required=True, type=str, help="Path to the bag file.")
@click.option(
    "--out_dir", required=True, type=str, help="Path to store extracted content to."
)
@click.option(
    "--include_camera",
    is_flag=True,
    help="Whether to extract camera images from the bag file.",
)
@click.option(
    "--include_lidar",
    is_flag=True,
    help="Whether to extract lidar point clouds from the bag file.",
)
@click.option(
    "--overwrite_existing",
    is_flag=True,
    help="Whether to overwrite existing content in the output directory. If not, then the extraction will be skipped for that file.",
)
def extract(
    bag_dir: str,
    out_dir: str,
    include_camera=False,
    include_lidar=False,
    overwrite_existing=False,
):
    assert (
        include_camera or include_lidar
    ), "At least one of camera or lidar must be included for extraction."

    timestamp_by_topic = defaultdict(list)
    counter_by_topic = defaultdict(int)
    short_names: dict[str, str] = {}
    total_messages = 0
    with Reader(bag_dir) as reader:
        topics = identify_topics(reader)
        camera_connections = topics.get("sensor_msgs/msg/CompressedImage", [])
        lidar_connections = topics.get("sensor_msgs/msg/PointCloud2", [])
        all_connections = (camera_connections if include_camera else []) + (
            lidar_connections if include_lidar else []
        )
        total_messages = sum([connection.msgcount for connection in all_connections])

        for connection in all_connections:
            short_names[connection.topic] = (
                infer_camera_name(connection.topic)
                if connection.msgtype == "sensor_msgs/msg/CompressedImage"
                else infer_lidar_name(connection.topic)
            )

        with tqdm.tqdm(desc="Extracting data.", total=total_messages) as pbar:
            for connection, timestamp, rawdata in reader.messages(all_connections):
                IS_CAMERA = connection.msgtype == "sensor_msgs/msg/CompressedImage"

                # Save content.
                if IS_CAMERA:
                    topic_dir = os.path.join(
                        out_dir, "camera", infer_camera_name(connection.topic)
                    )
                    output_path = os.path.join(
                        topic_dir,
                        f"image_{counter_by_topic[connection.topic]:05d}.png",
                    )
                    if not os.path.exists(topic_dir):
                        os.makedirs(topic_dir)

                    if not os.path.exists(output_path) or overwrite_existing:
                        cv2.imwrite(
                            output_path,
                            compressed_image_message_to_numpy(rawdata, connection),
                        )
                else:
                    topic_dir = os.path.join(
                        out_dir, "lidar", infer_lidar_name(connection.topic)
                    )
                    output_path = os.path.join(
                        topic_dir,
                        f"pointcloud_{counter_by_topic[connection.topic]:05d}.pkl",
                    )
                    if not os.path.exists(topic_dir):
                        os.makedirs(topic_dir)

                    if not os.path.exists(output_path) or overwrite_existing:
                        data = ros_lidar_to_numpy(
                            deserialize_cdr(rawdata, connection.msgtype)
                        )
                        with open(output_path, "wb") as f:
                            pickle.dump(data, f)

                timestamp_by_topic[connection.topic].append(timestamp)
                counter_by_topic[connection.topic] += 1
                pbar.update(1)

        with open(f"{out_dir}/timestamps.json", "w") as f:
            json.dump(
                {
                    short_names[topic]: timestamps
                    for (topic, timestamps) in timestamp_by_topic
                },
                f,
            )


if __name__ == "__main__":
    extract()
