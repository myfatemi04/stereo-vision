# Because of misalignment between sampling rates,
# this script provides the indices that correspond to unix timestamps
# for each topic.

import numpy as np
import json
import os
import sys

base_cameras_dir = 'M-MULTI-SLOW-KAIST_images'
base_lidar_dir = 'M-MULTI-SLOW-KAIST_lidar'

with open("./bags/extracted/M-MULTI-SLOW-KAIST/camera/timestamps.json", "r") as f:
    camera_ts = json.load(f)
            
with open("./bags/extracted/M-MULTI-SLOW-KAIST/lidar/timestamps.json", "r") as f:
    lidar_ts = json.load(f)

bag_start_timestamp = min(
    min(min(x) for x in camera_ts.values()),
    min(min(x) for x in lidar_ts.values())
)

# target_timestamp = camera_ts['front_left_center'][995]
if len(sys.argv) < 2:
    print("Please provide a timestamp. Usage: python timestamp_to_indexes.py <timestamp:seconds>")
    sys.exit(1)
else:
    target_timestamp = bag_start_timestamp + float(sys.argv[1]) * 1000000000

for camera in camera_ts:
    closest_index = -1
    closest_difference = 1e10
    for index in range(len(camera_ts[camera])):
        obs_timestamp = camera_ts[camera][index]
        difference = (obs_timestamp - target_timestamp)/1e9
        if abs(difference) < closest_difference:
            closest_index = index
            closest_difference = abs(difference)
        else:
            break
    
    print("Camera:", camera, "Index: ", closest_index)

for lidar in lidar_ts:
    closest_index = -1
    closest_difference = 1e10
    for index in range(len(lidar_ts[lidar])):
        obs_timestamp = lidar_ts[lidar][index]
        difference = (obs_timestamp - target_timestamp)/1e9
        if abs(difference) < closest_difference:
            closest_index = index
            closest_difference = abs(difference)
        else:
            break
    
    print("Lidar:", lidar, "Index: ", closest_index)
