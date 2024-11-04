"""
This script allows you to calculate which image or pkl file is the closest to a given timestamp.
"""

import json
import os

import click

SECONDS_TO_NANOSECONDS = 1_000_000_000


@click.command()
@click.option(
    "--bag_offset", "-o", required=True, type=float, help="Timestamp in seconds."
)
@click.option("--bag_dir", required=True, type=str, help="Path to the bag file.")
def get_timestamp(target_timestamp: float, bag_dir: str):
    with open(os.path.join(bag_dir, "camera/timestamps.json"), "r") as f:
        camera_timestamps: dict = json.load(f)

    with open(os.path.join(bag_dir, "lidar/timestamps.json"), "r") as f:
        lidar_timestamps: dict = json.load(f)

    bag_start_timestamp = min(
        min(x) for x in [*camera_timestamps.values(), *lidar_timestamps.values()]
    )
    target_timestamp = bag_start_timestamp + target_timestamp * SECONDS_TO_NANOSECONDS

    for label, ts in [("Camera", camera_timestamps), ("Lidar", lidar_timestamps)]:
        for device in ts.keys():
            closest_index = -1
            closest_difference = 1e10
            for index in range(len(ts[device])):
                obs_timestamp = ts[device][index]
                difference = (obs_timestamp - target_timestamp) / 1e9
                if abs(difference) < closest_difference:
                    closest_index = index
                    closest_difference = abs(difference)
                else:
                    break

        print(f"{label}: {device=}, {closest_index=}")
