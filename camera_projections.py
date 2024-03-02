import pickle
import numpy as np
import yaml
import matplotlib.pyplot as plt
import cv2
import json
from matrices import create_lidar_extrinsics, get_camera_extrinsics, expand_intrinsics
from matplotlib.widgets import Button, Slider
from functools import partial

# good timestamp: front_left_center#995

# Now, we should be able to convert from world space to camera space.
def test_camera_projection():
    # cameras_to_use = ['front_left', 'front_left_center', 'front_right']
    cameras_to_use = ['front_right']
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
    with open("calibration/intrinsic_models.json", "r") as f:
        data = json.load(f)
        distortion_coefficients = {k: np.array(v) for k, v in data['distortions'].items()}

    extrinsics = get_camera_extrinsics(use_yaml=False)

    fig = plt.figure(figsize=(len(cameras_to_use) * 4, 4))
    axes = [fig.add_subplot(1, 3, i + 1) for i in range(len(cameras_to_use))]

    # Slider for Alpha
    def update_alpha(new_alpha):
        nonlocal alpha
        alpha = new_alpha
        plot()

    tweak_params = {
        'x': 2.184,
        'y': -0.171,
        'z': 0.422,
        'yaw': 0.9599310886,
        'pitch': 0,
        'roll': 0,
    }
    # Assume this is fixed.
    tweak_xyz = np.array([2.184, -0.171, 0.422])
    camera_to_tweak = 'front_right'

    def tweak(field, new_value):
        from matrices import create_camera_extrinsics

        tweak_params[field] = new_value
        extrinsics[camera_to_tweak] = create_camera_extrinsics(
            tweak_params['x'],
            tweak_params['y'],
            tweak_params['z'],
            tweak_params['yaw'],
            tweak_params['pitch'],
            tweak_params['roll'],
        )
        plot()

    alpha = 0.1
    slider_axes = plt.axes([0.1, 0.01, 0.8, 0.02])
    slider = Slider(
        slider_axes,
        label='Alpha',
        valmin=0.0,
        valmax=1.0,
        valinit=alpha,
    )
    slider.on_changed(update_alpha)

    tweak_sliders = {}

    for i, param_name in enumerate(tweak_params.keys()):
        pitch_axes = plt.axes([0.1, 0.05 + 0.05 * (len(tweak_params) - i), 0.8, 0.02])
        minmax = (-np.pi/2, np.pi/2) if param_name in ['pitch', 'roll', 'yaw'] else [-3, 3]
        tweak_sliders[param_name] = Slider(
            pitch_axes,
            label=param_name.title(),
            valmin=minmax[0],
            valmax=minmax[1],
            valinit=tweak_params[param_name],
        )
        tweak_sliders[param_name].on_changed(partial(tweak, param_name))

    redistort = True

    def plot():
        for axis in axes:
            axis.clear()
        for i, camera in enumerate(cameras_to_use):
            intrinsics_expanded = expand_intrinsics(intrinsics[camera])
            camera_matrix = intrinsics_expanded @ extrinsics[camera]
            projected_points = (camera_matrix @ xyz.T).T
            projected_points = (projected_points / projected_points[:, [3]])[:, :3]
            Z = projected_points[:, 2]
            mask = Z > 0
            Z = Z[mask]
            projected_points = (projected_points / projected_points[:, [2]])[mask, :2]
            
            if redistort:
                points_in_camera_frame = (extrinsics[camera] @ xyz[mask].T).T[:, :3]
                cv2.projectPoints(
                    points_in_camera_frame,
                    np.zeros(3), # rvec
                    np.zeros(3), # tvec
                    intrinsics[camera],
                    distortion_coefficients[camera],
                )[0]
                # # during projection, distort the points.
                # projected_points = (extrinsics[camera] @ xyz[mask].T).T
                # projected_points = (projected_points / projected_points[:, [3]])[:, :2]
                # translated_distortion_coefficients = np.array([
                #     distortion_coefficients[camera][0],
                #     distortion_coefficients[camera][1],
                #     0,
                #     0
                # ])
                # print(projected_points.shape)
                # redistorted = cv2.fisheye.distortPoints(
                #     projected_points[np.newaxis, ...],
                #     intrinsics[camera],
                #     translated_distortion_coefficients,
                # )[0]
                # projected_points = redistorted

            axis = axes[i]
            axis.imshow(plt.imread(f"{base_path}/camera/{camera}/image_{index_camera[i]}.png")[::-1, :, :])
            # if 0 < x < 2064 and 0 < y < 960:
            axis.scatter(projected_points.T[0], projected_points.T[1], c=1/(Z+100), s=10, alpha=alpha, cmap='turbo')
            axis.set_xlim(0, 2064)
            axis.set_ylim(0, 960)
            axis.set_title(camera + " + LiDAR")
            # plt.scatter(
            #     points_camera_plane_2d[:, 0],
            #     points_camera_plane_2d[:, 1],
            #     c=-1/depths,
            #     s=1,
            #     cmap='jet',
            #     alpha=0.5,
            # )
            axis.axis('off')
            # if i == len(cameras_to_use) - 1:
            #     axis.colorbar()
            # axis.subplot(2, 3, i + 4)
            # axis.title(camera)
            # axis.imshow(plt.imread(f"{base_path}/camera/{camera}/image_{index_camera[i]}.png"))
            # axis.axis('off')

    plot()

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
