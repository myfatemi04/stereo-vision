import pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt
import cv2
import json

def create_extrinsics(x, y, z, yaw):
    """
    An extrinsic camera matrix follows the following format:
        [R t]
    where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
    """
    # 1. translate by -x, -y, -z
    tra_mat = np.array([
        [1, 0, 0, -x],
        [0, 1, 0, -y],
        [0, 0, 1, -z],
        [0, 0, 0, 1],
    ])
    # 2. rotate by -yaw
    rot_mat = np.array([
        [np.cos(-yaw), -np.sin(-yaw), 0, 0],
        [np.sin(-yaw), np.cos(-yaw), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    # 3. permute axes: x -> z, y -> x, z -> y
    fix_order = np.array([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ])
    Rt = fix_order @ rot_mat @ tra_mat

    # t = np.array([x, y, z])
    # R = np.array([
    #     [np.cos(yaw), -np.sin(yaw), 0],
    #     [np.sin(yaw), np.cos(yaw), 0],
    #     [0, 0, 1]
    # ])
    # Rt = np.concatenate((R, t.reshape(3, 1)), axis=1)
    # Derived from `plot_camera` function
    # fix_order = np.array([
    #     [0, 1, 0, 0],
    #     [0, 0, 1, 0],
    #     [1, 0, 0, 0],
    #     [0, 0, 0, 1],
    # ])
    # Rt = np.matmul(Rt, fix_order)

    return Rt

def load_extrinsics(use_yaml):
    if use_yaml:
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
    return extrinsics

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
    index = 1

    with open(f"M-MULTI-SLOW-KAIST_lidar/luminar_front_points/{index}.pkl", "rb") as f:
        xyz_front = pickle.load(f)[0]
    
    with open(f"M-MULTI-SLOW-KAIST_lidar/luminar_left_points/{index}.pkl", "rb") as f:
        xyz_left = pickle.load(f)[0]

    with open(f"M-MULTI-SLOW-KAIST_lidar/luminar_right_points/{index}.pkl", "rb") as f:
        xyz_right = pickle.load(f)[0]

    xyz = xyz_front # np.concatenate((xyz_front, xyz_left, xyz_right), axis=0)
    points = np.concatenate((xyz, np.ones((xyz.shape[0], 1))), axis=1)

    with open("intrinsic_models.json", "r") as f:
        intrinsic_models = json.load(f)
        intrinsic_matrices = {k: np.array(v) for k, v in intrinsic_models['matrices'].items()}
        distortion_coefficients = {k: np.array(v) for k, v in intrinsic_models['distortions'].items()}
    use_yaml = False
    extrinsics = load_extrinsics(use_yaml)
    if use_yaml:
        extrinsics = {k: invert_extrinsics(v) for k, v in extrinsics.items()}
    
    undistort = False

    plt.figure(figsize=(len(cameras_to_use) * 4, 4))
    for i, camera in enumerate(cameras_to_use):
        points_camera_frame_3d = np.matmul(invert_extrinsics(extrinsics[camera]), points.T).T
        points_camera_plane_2d = np.matmul(intrinsic_matrices[camera], points_camera_frame_3d.T).T
        depths = np.linalg.norm(points_camera_frame_3d, axis=1)
        # filter out points that end up behind the camera
        mask = points_camera_plane_2d[:, -1] > 0
        points_camera_plane_2d = points_camera_plane_2d[mask]
        depths = depths[mask]
        points_camera_plane_2d = (points_camera_plane_2d / points_camera_plane_2d[:, [-1]])
        # undistort
        if undistort:
            D = distortion_coefficients[camera]
            points_camera_plane_2d = cv2.undistortPoints(np.expand_dims(points_camera_plane_2d[:, :2], axis=0), np.eye(3), D).squeeze()
        else:
            points_camera_plane_2d = points_camera_plane_2d[:, :2]
        # filter out points that are out of frame
        x, y = points_camera_plane_2d.T
        mask = (0 < x) & (x < 2064) & (0 < y) & (y < 960)
        points_camera_plane_2d = points_camera_plane_2d[mask]
        depths = depths[mask]

        plt.subplot(2, 3, i + 1)
        plt.imshow(plt.imread(f"{base_path}/{camera}/{index}.png"))
        # if 0 < x < 2064 and 0 < y < 960:
        plt.scatter(
            points_camera_plane_2d[:, 0],
            points_camera_plane_2d[:, 1],
            c=-1/depths,
            s=0.1,
            cmap='jet',
            alpha=0.5,
        )
        plt.axis('off')
        if i == len(cameras_to_use) - 1:
            plt.colorbar()
        plt.subplot(2, 3, i + 4)
        plt.title(camera)
        plt.imshow(plt.imread(f"{base_path}/{camera}/{index}.png"))
        plt.axis('off')

    plt.savefig("camera_projections.png")

def test_camera_projection_opencv():
    # cameras_to_use = ['front_right']
    cameras_to_use = ['front_left', 'front_left_center', 'front_right']
    base_path = 'M-MULTI-SLOW-KAIST_images'
    index = 1

    with open(f"M-MULTI-SLOW-KAIST_lidar/luminar_front_points/{index}.pkl", "rb") as f:
        xyz_front = pickle.load(f)[0]
    
    with open(f"M-MULTI-SLOW-KAIST_lidar/luminar_left_points/{index}.pkl", "rb") as f:
        xyz_left = pickle.load(f)[0]

    with open(f"M-MULTI-SLOW-KAIST_lidar/luminar_right_points/{index}.pkl", "rb") as f:
        xyz_right = pickle.load(f)[0]

    xyz = xyz_front # np.concatenate((xyz_front, xyz_left, xyz_right), axis=0)
    points = np.ascontiguousarray(xyz)
    # points = np.concatenate((xyz, np.ones((xyz.shape[0], 1))), axis=1)

    extrinsics = load_extrinsics(use_yaml=True)
    with open("intrinsic_models.json", "r") as f:
        intrinsic_models = json.load(f)
        intrinsic_matrices = {k: np.array(v) for k, v in intrinsic_models['matrices'].items()}
        distortion_coefficients = {k: np.array(v) for k, v in intrinsic_models['distortions'].items()}

    plt.figure(figsize=(4 * len(cameras_to_use) * 1.5, 4))
    for i, camera in enumerate(cameras_to_use):
        R = extrinsics[camera][:3, :3]
        t = extrinsics[camera][:3, 3]
        K = intrinsic_matrices[camera]
        D = distortion_coefficients[camera]

        # proj_mat = np.dot(K, np.hstack((R, t[:, np.newaxis])))
        # # convert 3D points into homgenous points
        # xyz_hom = np.hstack((points, np.ones((points.shape[0], 1))))

        # xy_hom = np.dot(proj_mat, xyz_hom.T).T

        # # get 2d coordinates in image [pixels]
        # z = xy_hom[:, -1]
        # xy = xy_hom[:, :2] / np.tile(z[:, np.newaxis], (1, 2))

        # # undistort - has to be 1xNx2 structure
        # xy = cv2.undistortPoints(np.expand_dims(xy, axis=0), np.eye(3), D).squeeze()

        # # drop all points behind camera
        # xy = xy[z > 0]

        mask = np.ones((points.shape[0]), dtype=bool)
        if 'front' in camera:
            mask = mask & (points[:, 0] > 0)
        if 'rear' in camera:
            mask = mask & (points[:, 0] < 0)
        if 'left' in camera:
            mask = mask & (points[:, 1] > 0)
        if 'right' in camera:
            mask = mask & (points[:, 1] < 0)
        points_masked = np.ascontiguousarray(points[mask][:, :3])
        # points = points[points[:, 1] < 0]
        xy, _ = cv2.projectPoints(points_masked, R, t, K, D)
        # filter out points that are out of frame
        xy = xy[:, 0]
        x, y = xy.T
        mask = (0 < x) & (x < 2064) & (0 < y) & (y < 960)
        xy = xy[mask]

        plt.subplot(1, len(cameras_to_use), i + 1)
        plt.imshow(plt.imread(f"{base_path}/{camera}/{index}.png"))
        # if 0 < x < 2064 and 0 < y < 960:
        plt.scatter(xy[:, 0], xy[:, 1], s=0.1, alpha=0.5) # , c=-1/depths, s=1, cmap='jet')
        # plt.scatter(points_camera_plane_2d[:, 0], points_camera_plane_2d[:, 1], c=-1/depths, s=1, cmap='jet')
        plt.axis('off')
        plt.title(camera)
        # if i == len(cameras_to_use) - 1:
        #     plt.colorbar()

    plt.savefig("camera_projections.png")

test_camera_projection_opencv()
# test_camera_projection()
