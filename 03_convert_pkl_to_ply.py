# Format of each .pkl file:
# (point_xyzs, point_intensities, point_rings, point_times)

import os
import pickle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tqdm
import open3d as o3d

from examples.matrices import create_lidar_extrinsics


def point_cloud_xyz_to_ply(xyz):
    return f"""ply
format ascii 1.0
element vertex {len(xyz)}
property float x
property float y
property float z
end_header
""" + "\n".join(
        f"{point[0]} {point[1]} {point[2]}" for point in xyz
    )


# These are w.r.t. rear_axle_middle_ground.
# They take points from, e.g., /luminar_front_points, and transform them into the frame for /luminar_points.
front_ext = create_lidar_extrinsics(2.342, 0, 0.448, 0)
left_ext = create_lidar_extrinsics(1.549, 0.267, 0.543, 2.0943951024)
right_ext = create_lidar_extrinsics(1.549, -0.267, 0.543, -2.0943951024)

input_path = "bags/extracted/M-MULTI-SLOW-KAIST/lidar/luminar_front_points"
output_path = f"{input_path}_rear_axle_middle_ground_ply_files"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Go through the directory, converting to .ply.
# Use a ThreadPoolExecutor because this is an IO-intensive task.
jobs = os.listdir(input_path)


def run_job(path):
    target = f"{output_path}/{path.replace('.pkl', '.ply')}"
    if os.path.exists(target):
        return

    with open(f"{input_path}/{path}", "rb") as f:
        xyz = pickle.load(f)[0]

    xyz = (front_ext @ np.concatenate((xyz, np.ones((len(xyz), 1))), axis=1).T).T
    # Remove homogenous coordinate
    xyz = xyz[:, :3]
    intensity = xyz[:, 3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.intensity = o3d.utility.Vector1dVector(intensity)
    o3d.io.write_point_cloud(target, pcd)


# with ThreadPoolExecutor() as executor:
#     list(tqdm.tqdm(executor.map(run_job, jobs), total=len(jobs), desc='Converting .pkl to .ply'))

import sys

index = int(sys.argv[1])
run_job(f"pcd_{index}.pkl")
os.system(f"cp {output_path}/pcd_{index}.ply pcd_{index}.ply")
