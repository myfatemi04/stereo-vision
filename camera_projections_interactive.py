import pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt
import cv2
import json
from matrices import create_lidar_extrinsics, get_camera_extrinsics, expand_intrinsics

# good timestamp: front_left_center#995

# Now, we should be able to convert from world space to camera space.
def test_camera_projection():
    cameras_to_use = ['front_left', 'front_left_center', 'front_right']
    base_path = 'bags/extracted/M-MULTI-SLOW-KAIST'

    front_ext = create_lidar_extrinsics(2.342, 0, 0.448, 0)
    left_ext = create_lidar_extrinsics(1.549, 0.267, 0.543, 2.0943951024)
    right_ext = create_lidar_extrinsics(1.549, -0.267, 0.543, -2.0943951024)

    index_lidar = (2339, 2079, 1948)
    index_camera = (585, 995, 595)
    with open(f"{base_path}/lidar/luminar_front_points/pcd_{index_lidar[0]}.pkl", "rb") as f:
        xyz_front = pickle.load(f)[0]
        xyz_front = (front_ext @ np.concatenate((xyz_front, np.ones((len(xyz_front), 1))), axis=1).T).T

    with open(f"{base_path}/lidar/luminar_left_points/pcd_{index_lidar[1]}.pkl", "rb") as f:
        xyz_left = pickle.load(f)[0]
        xyz_left = (left_ext @ np.concatenate((xyz_left, np.ones((len(xyz_left), 1))), axis=1).T).T

    with open(f"{base_path}/lidar/luminar_right_points/pcd_{index_lidar[2]}.pkl", "rb") as f:
        xyz_right = pickle.load(f)[0]
        xyz_right = (right_ext @ np.concatenate((xyz_right, np.ones((len(xyz_right), 1))), axis=1).T).T

    xyz = np.concatenate((xyz_front, xyz_left, xyz_right), axis=0)
    xyz[:, 1] *= -1

    with open("calibration/intrinsic_matrices.json", "r") as f:
        intrinsics = json.load(f)
        intrinsics = {k: np.array(v) for k, v in intrinsics.items()}

    extrinsics = get_camera_extrinsics(use_yaml=False)
    
    undistort = False

    plt.figure(figsize=(len(cameras_to_use) * 4, 4))
    for i, camera in enumerate(cameras_to_use):
        intrinsics_expanded = expand_intrinsics(intrinsics[camera])
        camera_matrix = intrinsics_expanded @ extrinsics[camera]
        projected_points = (camera_matrix @ np.array(xyz).T).T
        projected_points = (projected_points / projected_points[:, [3]])[:, :3]
        Z = projected_points[:, 2]
        mask = Z > 0
        Z = Z[mask]
        projected_points = (projected_points / projected_points[:, [2]])[mask, :2]

        # points_camera_frame_3d = np.matmul(extrinsics[camera], points.T).T
        # points_camera_plane_2d = np.matmul(expand_intrinsics(intrinsics[camera]), points_camera_frame_3d.T).T
        # depths = np.linalg.norm(points_camera_frame_3d, axis=1)
        # # filter out points that end up behind the camera
        # mask = points_camera_plane_2d[:, -1] > 0
        # points_camera_plane_2d = points_camera_plane_2d[mask]
        # depths = depths[mask]
        # points_camera_plane_2d = (points_camera_plane_2d / points_camera_plane_2d[:, [-1]])
        # undistort
        # if undistort:
        #     D = distortion_coefficients[camera]
        #     points_camera_plane_2d = cv2.undistortPoints(np.expand_dims(points_camera_plane_2d[:, :2], axis=0), np.eye(3), D).squeeze()
        # else:
        # points_camera_plane_2d = points_camera_plane_2d[:, :2]
        # filter out points that are out of frame
        # x, y = points_camera_plane_2d.T
        # mask = (0 < x) & (x < 2064) & (0 < y) & (y < 960)
        # points_camera_plane_2d = points_camera_plane_2d[mask]
        # depths = depths[mask]

        plt.subplot(2, 3, i + 1)
        plt.imshow(plt.imread(f"{base_path}/camera/{camera}/image_{index_camera[i]}.png")[::-1, :, :])
        # if 0 < x < 2064 and 0 < y < 960:
        plt.scatter(projected_points.T[0], projected_points.T[1], c=1/(Z+100), s=10, alpha=0.5, cmap='turbo')
        plt.xlim(0, 2064)
        plt.ylim(0, 960)
        plt.title(camera + " + LiDAR")
        # plt.scatter(
        #     points_camera_plane_2d[:, 0],
        #     points_camera_plane_2d[:, 1],
        #     c=-1/depths,
        #     s=1,
        #     cmap='jet',
        #     alpha=0.5,
        # )
        plt.axis('off')
        if i == len(cameras_to_use) - 1:
            plt.colorbar()
        plt.subplot(2, 3, i + 4)
        plt.title(camera)
        plt.imshow(plt.imread(f"{base_path}/camera/{camera}/image_{index_camera[i]}.png"))
        plt.axis('off')

    # plt.savefig("camera_projections.png")
    plt.show()

def test_camera_projection_opencv():
    # cameras_to_use = ['front_right']
    cameras_to_use = ['front_left', 'front_left_center', 'front_right']
    base_path = 'M-MULTI-SLOW-KAIST_images'
    index = 1

    front_ext = create_lidar_extrinsics(2.342, 0, 0.448, 0)
    left_ext = create_lidar_extrinsics(1.549, 0.267, 0.543, 2.0943951024)
    right_ext = create_lidar_extrinsics(1.549, -0.267, 0.543, -2.0943951024)

    index = 1
    with open(f"M-MULTI-SLOW-KAIST_lidar/luminar_front_points/{index}.pkl", "rb") as f:
        xyz_front = pickle.load(f)[0]
        xyz_front = (front_ext @ np.concatenate((xyz_front, np.ones((len(xyz_front), 1))), axis=1).T).T

    with open(f"M-MULTI-SLOW-KAIST_lidar/luminar_left_points/{index}.pkl", "rb") as f:
        xyz_left = pickle.load(f)[0]
        xyz_left = (left_ext @ np.concatenate((xyz_left, np.ones((len(xyz_left), 1))), axis=1).T).T

    with open(f"M-MULTI-SLOW-KAIST_lidar/luminar_right_points/{index}.pkl", "rb") as f:
        xyz_right = pickle.load(f)[0]
        xyz_right = (right_ext @ np.concatenate((xyz_right, np.ones((len(xyz_right), 1))), axis=1).T).T

    xyz = np.concatenate((xyz_front, xyz_left, xyz_right), axis=0)
    points = np.ascontiguousarray(xyz)
    # points = np.concatenate((xyz, np.ones((xyz.shape[0], 1))), axis=1)

    extrinsics = get_camera_extrinsics(use_yaml=True)
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

# test_camera_projection_opencv()
test_camera_projection()
