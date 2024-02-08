# Because of misalignment between sampling rates,
# this script provides the indices that correspond to unix timestamps
# for each topic.

import numpy as np
import json
import os

base_cameras_dir = 'M-MULTI-SLOW-KAIST_images'
base_lidar_dir = 'M-MULTI-SLOW-KAIST_lidar'

camera_ts = {
    'front_left_center': [],
    'front_right': [],
    'front_left': [],
    'rear_left': [],
    'rear_right': []
}

earliest_timestamp = None

# for camera in ['front_left_center']:
#     with open(os.path.join(base_cameras_dir, camera, "timestamps.json"), "r") as f:
#         camera_ts[camera] = json.load(f)

#         if earliest_timestamp is None or camera_ts[camera][0] < earliest_timestamp:
#             earliest_timestamp = camera_ts[camera][0]

with open("./bags/extracted/M-MULTI-SLOW-KAIST/camera/timestamps.json", "r") as f:
    camera_ts = json.load(f)

# lidar_ts = {
#     'luminar_left_points': [],
#     'luminar_right_points': [],
#     'luminar_front_points': [],
# }
            
with open("./bags/extracted/M-MULTI-SLOW-KAIST/lidar/timestamps.json", "r") as f:
    lidar_ts = json.load(f)

# for lidar in ['luminar_front_points', 'luminar_left_points', 'luminar_right_points']:
#     with open(os.path.join(base_lidar_dir, lidar, "timestamps.json"), "r") as f:
#         lidar_ts[lidar] = json.load(f)

# Look into front_left_center/255
target_timestamp = camera_ts['front_left_center'][995]

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
    closest_index = 0
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
