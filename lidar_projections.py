import pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt
import cv2
import json
from examples.matrices import (
    create_lidar_extrinsics,
    get_camera_extrinsics,
    expand_intrinsics,
)
from matplotlib.colors import rgb2hex

# good timestamp: front_left_center#995


# Now, we should be able to convert from world space to camera space.
def get_rgbd():
    cameras_to_use = ["front_left", "front_left_center", "front_right"]
    base_path = "bags/extracted/M-MULTI-SLOW-KAIST"

    front_ext = create_lidar_extrinsics(2.342, 0, 0.448, 0)
    left_ext = create_lidar_extrinsics(1.549, 0.267, 0.543, 2.0943951024)
    right_ext = create_lidar_extrinsics(1.549, -0.267, 0.543, -2.0943951024)

    index_lidar = (2339, 2079, 1948)
    index_camera = (585, 995, 595)
    with open(
        f"{base_path}/lidar/luminar_front_points/{index_lidar[0]}.pkl", "rb"
    ) as f:
        xyz_front = pickle.load(f)[0]
        xyz_front = (
            front_ext
            @ np.concatenate((xyz_front, np.ones((len(xyz_front), 1))), axis=1).T
        ).T

    with open(f"{base_path}/lidar/luminar_left_points/{index_lidar[1]}.pkl", "rb") as f:
        xyz_left = pickle.load(f)[0]
        xyz_left = (
            left_ext @ np.concatenate((xyz_left, np.ones((len(xyz_left), 1))), axis=1).T
        ).T

    with open(
        f"{base_path}/lidar/luminar_right_points/{index_lidar[2]}.pkl", "rb"
    ) as f:
        xyz_right = pickle.load(f)[0]
        xyz_right = (
            right_ext
            @ np.concatenate((xyz_right, np.ones((len(xyz_right), 1))), axis=1).T
        ).T

    xyz = np.concatenate((xyz_front, xyz_left, xyz_right), axis=0)
    xyz[:, 1] *= -1

    with open("calibration/intrinsic_matrices.json", "r") as f:
        intrinsics = json.load(f)
        intrinsics = {k: np.array(v) for k, v in intrinsics.items()}

    extrinsics = get_camera_extrinsics()

    undistort = False

    points = []
    examined = set()

    for i, camera in enumerate(["front_left_center"]):
        intrinsics_expanded = expand_intrinsics(intrinsics[camera])
        camera_matrix = intrinsics_expanded @ extrinsics[camera]

        imgrgb = plt.imread(f"{base_path}/camera/{camera}/{index_camera[i]}.png")
        for point in np.array(xyz):
            projected_point = (camera_matrix @ point.T).T
            projected_point = (projected_point / projected_point[3])[:3]
            Z = projected_point[2]
            if Z > 0:
                continue
            projected_point = (projected_point / projected_point[2])[:2]
            if 0 < projected_point[0] < 2064 and 0 < projected_point[1] < 960:
                examined.add(str(point))
                x, y = int(point[0]), int(point[1])
                pixel_color = imgrgb[y, x]
                hex_color = rgb2hex(pixel_color)
                points.append((point[0], point[1], point[2], hex_color))

    # Extract x, y, z coordinates, and colors from the data
    x = [item[0] for item in points]
    y = [item[1] for item in points]
    z = [item[2] for item in points]
    colors = [item[3] for item in points]

    # Create 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    xyz = np.array([x for x in xyz if str(x) not in examined])
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c="pink", s=5, alpha=0.1)
    ax.scatter(x, y, z, c=colors, s=10)

    # Add labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Scatter Plot with Hex Colors")

    # Show plot
    plt.show()


get_rgbd()
