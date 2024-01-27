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

def create_camera_matrix(extrinsics, intrinsics):
    # Get 3x3 rotation matrix
    R = extrinsics[:3, :3]
    # Get 3x1 translation matrix
    C = np.expand_dims(extrinsics[:3, 3], 0).T
    # inverse of rotation matrix is transpose, as dimensions are orthonormal.
    # Invert C
    R_inv = R.T
    R_inv_C = np.matmul(R_inv, C)
    # Create new matrix
    extrinsics = np.concatenate((R_inv, -R_inv_C), -1)
    # Multiply matrices to form camera projection matrix
    cam_proj_mat = np.matmul(intrinsics, extrinsics)
    return cam_proj_mat

# Now, we should be able to convert from world space to camera space.
def test_camera_projection():
    cameras_to_use = ['front_left', 'front_left_center', 'front_right']
    base_path = 'M-MULTI-SLOW-KAIST_images'
    index = 1
    X, Y, Z = np.meshgrid(np.linspace(0, 1, 8), np.linspace(0, 1, 8), np.linspace(0, 1, 8))
    points = np.stack((X.flatten(), Y.flatten(), Z.flatten(), np.ones(X.flatten().shape[0])), axis=1)
    points[:, 0] += 10
    points[:, 1] += -0.5

    for i, camera in enumerate(cameras_to_use):
        mat = create_camera_matrix(extrinsics[camera], intrinsics[camera])
        point_camera_space = (mat @ points.T)
        # filter out points that end up behind the camera
        point_camera_space = point_camera_space[:, point_camera_space[-1] > 0]
        point_camera_space = (point_camera_space / point_camera_space[-1])

        plt.subplot(1, 3, i + 1)
        plt.imshow(plt.imread(f"{base_path}/{camera}/{index}.png"))
        # if 0 < x < 2064 and 0 < y < 960:
        plt.scatter(point_camera_space[0], point_camera_space[1])
        plt.axis('off')
        plt.title(camera)

    plt.savefig("camera_projections.png")

test_camera_projection()
