import pickle

import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d import Axes3D

ZDIR = 'x'

def plot_camera(ax: Axes3D, extrinsic, intrinsic, color):
    # Plot camera frame axes
    tx, ty, tz = extrinsic[0, 3], extrinsic[1, 3], extrinsic[2, 3]
    # ax.quiver(tx, ty, tz,
    #           extrinsic[0, 0], extrinsic[1, 0], extrinsic[2, 0],
    #           color=color[0], arrow_length_ratio=0.1)
    # ax.quiver(tx, ty, tz,
    #           extrinsic[0, 1], extrinsic[1, 1], extrinsic[2, 1],
    #           color=color[1], arrow_length_ratio=0.1)
    # ax.quiver(tx, ty, tz,
    #           extrinsic[0, 2], extrinsic[1, 2], extrinsic[2, 2],
    #           color=color[2], arrow_length_ratio=0.1)

    ax.plot([tx, tx + extrinsic[0, 0]], [ty, ty + extrinsic[1, 0]], [tz, tz + extrinsic[2, 0]], color=color[0], zdir=ZDIR)
    ax.plot([tx, tx + extrinsic[0, 1]], [ty, ty + extrinsic[1, 1]], [tz, tz + extrinsic[2, 1]], color=color[1], zdir=ZDIR)
    ax.plot([tx, tx + extrinsic[0, 2]], [ty, ty + extrinsic[1, 2]], [tz, tz + extrinsic[2, 2]], color=color[2], zdir=ZDIR)

    if intrinsic is None:
        return
    
    # image bounds
    width = 2064
    height = 960
    pts = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [width, height, 1],
        [0, height, 1]
    ])
    ## convert to homogenous
    ii = np.linalg.inv(intrinsic)
    pts_camera_3d = np.matmul(ii, pts.T).T
    pts_camera_3d *= 4
    pts_camera_3d = np.concatenate((pts_camera_3d, np.ones((4, 1))), axis=1)
    fix_order = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
    """
    This "fix_order" matrix was found by experimenting with the extrinsic
    and intrinsic axes. I found that this step was necessary before converting
    to world coordinates with the extrinsic matrix.
    pts_camera_3d = pts_camera_3d[:, [2, 0, 1, 3]]
    """
    pts_camera_3d = np.matmul(fix_order, pts_camera_3d.T).T
    pts_world_3d = np.matmul(extrinsic, pts_camera_3d.T).T
    for i in range(4):
        x, y, z, _ = pts_world_3d[i]
        ax.plot([tx, x], [ty, y], [tz, z], color='black', zdir=ZDIR)
    # plot connectors for image bounds
    # add the first point to the end to close the loop
    x = [*pts_world_3d[:, 0], pts_world_3d[0, 0]]
    y = [*pts_world_3d[:, 1], pts_world_3d[0, 1]]
    z = [*pts_world_3d[:, 2], pts_world_3d[0, 2]]
    ax.plot(x, y, z, color='black', zdir=ZDIR)

def create_extrinsics(x, y, z, yaw, is_camera=True):
    """
    An extrinsic camera matrix follows the following format:
        [R t]
    where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
    """
    # 1. translate
    if is_camera:
        tra = np.array([
            [1, 0, 0, -x],
            [0, 1, 0, -y],
            [0, 0, 1, -z],
            [0, 0, 0, 1],
        ])
    else:
        tra = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ])
    # 2. rotate
    if is_camera:
        yaw = -yaw
    rot = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw), np.cos(yaw), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    # 3. permute axes: x -> z, y -> x, z -> y
    if is_camera:
        permute = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ])
        Rt = permute @ rot @ tra
    else:
        Rt = rot @ tra
    return Rt

def invert_extrinsics(extrinsics):
    """
    The extrinsic matrix is 3x4. Therefore, we can't simply take a matrix inverse.
    We need to inver the rotation matrix and translation matrix separately.
    Specifically, we:
    (1) invert the rotation matrix, and
    (2) multiply the translation matrix by the inverse rotation matrix, and
    (3) invert the translation by negating the translation matrix
    """
    # Get 3x3 rotation matrix
    R = extrinsics[:3, :3]
    # Get 3x1 translation matrix
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    # inverse of rotation matrix is transpose, as dimensions are orthonormal.
    # Invert C
    R_inv = R.T
    R_inv_C = np.matmul(R_inv, C)
    # Create new matrix
    return np.concatenate((R_inv, -R_inv_C), -1)

# Load extrinsics
USE_YAML = False
if USE_YAML:
    extrinsics = {}
    camera_names = ['front_left', 'front_right', 'rear_left', 'rear_right', 'front_left_center', 'front_right_center']

    for camera in camera_names:
        with open(f"extrinsics/{camera}.yaml", "r") as f:
            data = yaml.load(f, yaml.BaseLoader)
            R = np.array(data['R']['data']).reshape(3, 3)
            t = np.array(data['T']['data']).reshape(3, 1)
            Rt = np.concatenate((R, t), axis=1).astype(np.float64)
            extrinsics[camera] = Rt # invert_extrinsics(Rt)
else:
    # These are from the URDF file in RACECAR
    extrinsics = {
        'front_left': create_extrinsics(2.184, 0.171, 0.422, -0.9599310886),
        'front_right': create_extrinsics(2.184, -0.171, 0.422, +0.9599310886),
        'rear_left': create_extrinsics(1.473, 0.140, 0.543, -2.2689280276),
        'rear_right': create_extrinsics(1.473, -0.140, 0.543, +2.2689280276),
        'front_left_center': create_extrinsics(2.235, 0.121, 0.422, 0),
        'front_right_center': create_extrinsics(2.235, -0.121, 0.422, 0),
    }

import json

with open("intrinsic_matrices.json", "r") as f:
    intrinsics = json.load(f)
    intrinsics = {k: np.array(v) for k, v in intrinsics.items()}

# Create a 3D plot
fig = plt.figure()
ax: Axes3D = fig.add_subplot(111, projection='3d') # type: ignore

# Color scheme (X, Y, Z) -> (R, G, B)
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

# Plot each camera
# for camera in extrinsics.keys():
#     plot_camera(ax, extrinsics[camera], None, colors)
for camera in intrinsics.keys():
    plot_camera(ax, extrinsics[camera], intrinsics[camera], colors)

front_ext = create_extrinsics(2.342, 0, 0.448, 0, is_camera=False)
left_ext = create_extrinsics(1.549, 0.267, 0.543, 2.0943951024, is_camera=False)
right_ext = create_extrinsics(1.549, -0.267, 0.543, -2.0943951024, is_camera=False)

index = 1
with open(f"front_{index}.pkl", "rb") as f:
    xyz_front = pickle.load(f)[0]
    xyz_front = (front_ext @ np.concatenate((xyz_front, np.ones((len(xyz_front), 1))), axis=1).T).T

with open(f"left_{index}.pkl", "rb") as f:
    xyz_left = pickle.load(f)[0]
    xyz_left = (left_ext @ np.concatenate((xyz_left, np.ones((len(xyz_left), 1))), axis=1).T).T

with open(f"right_{index}.pkl", "rb") as f:
    xyz_right = pickle.load(f)[0]
    xyz_right = (right_ext @ np.concatenate((xyz_right, np.ones((len(xyz_right), 1))), axis=1).T).T

xyz = np.concatenate((xyz_front, xyz_left, xyz_right), axis=0)
xyz = xyz[:, :3]
# their luminar points are uncalibrated
# xyz = xyz_front
# xyz = xyz_left
xyz = xyz[::10]
xyz1 = np.concatenate((xyz, np.ones((len(xyz), 1))), axis=1)
xyz = (extrinsics['front_right'] @ xyz1.T).T[:, :3]
ax.scatter(*xyz.T, s=1, zdir='x') # type: ignore

# its = intrinsics['front_right']
# xyh = (its @ xyz.T).T
# xyh = xyz
# xy = xyh[:, :2] / xyh[:, -1:]
# ax.scatter(xy[:, 0], xy[:, 1], 1)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set plot limits
size = 10
ax.set_xlim(-size, size)
ax.set_ylim(-size, size)
ax.set_zlim(-size, size)

# Show the plot
plt.show()
