import pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt

def create_extrinsics(x, y, z, yaw):
    """
    An extrinsic camera matrix follows the following format:
        [R t]
    where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
    """
    t = np.array([x, y, z])
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    Rt = np.concatenate((R, t.reshape(3, 1)), axis=1)
    # Derived from `plot_camera` function
    fix_order = np.array([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])
    Rt = np.matmul(Rt, fix_order)

    return Rt

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
            extrinsics[camera] = Rt
else:
    # These are from the URDF file in RACECAR
    extrinsics = {
        'front_left': create_extrinsics(2.184, 0.171, 0.422, -0.9599310886),
        'front_right': create_extrinsics(2.184, -0.171, 0.422, 0.9599310886),
        'rear_left': create_extrinsics(1.473, 0.140, 0.543, -2.2689280276),
        'rear_right': create_extrinsics(1.473, -0.140, 0.543, 2.2689280276),
        'front_left_center': create_extrinsics(2.235, 0.121, 0.422, 0),
        'front_right_center': create_extrinsics(2.235, -0.121, 0.422, 0),
    }
    # extrinsics = {
    #     'front_left': create_extrinsics(2.184, 0.171, 0.422, 0.9599310886),
    #     'front_right': create_extrinsics(2.184, -0.171, 0.422, -0.9599310886),
    #     'rear_left': create_extrinsics(1.473, 0.140, 0.543, 2.2689280276),
    #     'rear_right': create_extrinsics(1.473, -0.140, 0.543, -2.2689280276),
    #     'front_left_center': create_extrinsics(2.235, 0.121, 0.422, 0),
    #     'front_right_center': create_extrinsics(2.235, -0.121, 0.422, 0),
    # }

import json
with open("intrinsic_matrices.json", "r") as f:
    intrinsics = json.load(f)
    intrinsics = {k: np.array(v) for k, v in intrinsics.items()}

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

def create_camera_matrix(extrinsics, intrinsics):
    return np.matmul(intrinsics, invert_extrinsics(extrinsics))

# Now, we should be able to convert from world space to camera space.
def test_camera_projection():
    cameras_to_use = ['front_left', 'front_left_center', 'front_right']
    base_path = 'M-MULTI-SLOW-KAIST_images'
    index = 100
    X, Y, Z = np.meshgrid(np.linspace(0, 1, 8), np.linspace(0, 1, 8), np.linspace(0, 1, 8))

    with open(f"M-MULTI-SLOW-KAIST_lidar/luminar_front_points/{index}.pkl", "rb") as f:
        data = pickle.load(f)
        X, Y, Z = data[0].T

    plt.figure(figsize=(12, 4))
    
    points = np.stack((X.flatten(), Y.flatten(), Z.flatten(), np.ones(X.flatten().shape[0])), axis=1)
    points[:, 0] += 10
    points[:, 1] += -0.5

    for i, camera in enumerate(cameras_to_use):
        points_camera_frame_3d = np.matmul(invert_extrinsics(extrinsics[camera]), points.T).T
        points_camera_plane_2d = np.matmul(intrinsics[camera], points_camera_frame_3d.T).T
        depths = np.linalg.norm(points_camera_frame_3d, axis=1)
        # filter out points that end up behind the camera
        mask = points_camera_plane_2d[:, -1] > 0
        points_camera_plane_2d = points_camera_plane_2d[mask]
        depths = depths[mask]
        points_camera_plane_2d = (points_camera_plane_2d / points_camera_plane_2d[:, [-1]])
        # filter out points that are out of frame
        x = points_camera_plane_2d[:, 0]
        y = points_camera_plane_2d[:, 1]
        mask = (0 < x) & (x < 2064) & (0 < y) & (y < 960)
        points_camera_plane_2d = points_camera_plane_2d[mask]
        depths = depths[mask]

        plt.subplot(1, 3, i + 1)
        plt.imshow(plt.imread(f"{base_path}/{camera}/{index}.png"))
        # if 0 < x < 2064 and 0 < y < 960:
        plt.scatter(points_camera_plane_2d[:, 0], points_camera_plane_2d[:, 1], c=-1/depths, s=1, cmap='jet')
        plt.axis('off')
        plt.title(camera)
        if i == len(cameras_to_use) - 1:
            plt.colorbar()

    plt.savefig("camera_projections.png")

test_camera_projection()
