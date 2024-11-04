#!/bin/sh

# camera_model: plumb_bob
# intrinsic: 3394.88261,3395.074896,981.607769,616.689835
# distortion: -0.193355,-0.178479,0.000871,-0.003759,0.0

export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

. /opt/ros/humble/setup.sh
. ~/DirectVisualLiDARCalibration/rosws/install/setup.sh

image=bags/extracted/M-MULTI-SLOW-KAIST/camera/front_left_center/image_2521.png
map=bags/extracted/M-MULTI-SLOW-KAIST/lidar/luminar_front_points_rear_axle_middle_ground_ply_files/pcd_5759.ply
dst=calib1

ros2 run direct_visual_lidar_calibration preprocess_map \
  --map_path $map \
  --image_path $image \
  --dst_path $dst \
  --camera_model plumb_bob \
  --camera_intrinsics 3394.88261,3395.074896,981.607769,616.689835 \
  --camera_distortion_coeffs -0.193355,-0.178479,0.000871,-0.003759,0.0

# Manually create guess
ros2 run direct_visual_lidar_calibration initial_guess_manual $dst

# Fine registration
ros2 run direct_visual_lidar_calibration calibrate $dst
