import pickle

import matplotlib.pyplot as plt
import numpy as np
import yaml
from mpl_toolkits.mplot3d import Axes3D
import json

# Color scheme (X, Y, Z) -> (R, G, B)
AXIS_COLORS = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

def plot_camera(ax: Axes3D, extrinsic, intrinsic=None):
    """
    Note that we did R @ T. Thus, to recover the original T from the extrinsic matrix,
    we must left-multiply the extrinsic matrix by R^-1. This is OK because we know that
    R = the top-left 3x3 of the original matrix, and R^-1 = R^T because R is orthonormal.
    """
    extrinsic_inverse = invert_extrinsics(extrinsic)
    tx, ty, tz = extrinsic_inverse[:3, 3]
    """
    Part I. Plotting the camera's position and orientation.

    Recall that each row represents what the camera is "detecting"
    So row 0 = the world-coordinates axis along which the camera's x axis
    takes a projection.
    """
    R = extrinsic[:3, :3]
    Rx = R[0]
    Ry = R[1]
    Rz = R[2]
    """
    Plot these in 3D. Start from (tx, ty, tz) and draw a line along the axis.
    """
    for color, axis_direction in zip(AXIS_COLORS, [Rx, Ry, Rz]):
        ax.plot([tx, tx + axis_direction[0]], [ty, ty + axis_direction[1]], [tz, tz + axis_direction[2]], color=color)

    """ Plotting the intrinsic matrix is optional. """
    if intrinsic is None:
        return
    
    """
    Part II. Plotting the camera's field of view. We will plot the points
    ({0, width}, {0, height}, 1)

    That is, the four corners of the screen, at a depth of 1 meter.
    Note that these are sort of in "pixel" space, which is generated
    from the camera frame by left-multiplying by the intrinsic matrix.

    The 1 at the end represents the Z value in camera space; this information
    is lost when projecting from 3D to 2D.
    """
    width = 2064
    height = 960
    points_in_pixel_space = np.array([
        [0,     0,      1],
        [width, 0,      1],
        [width, height, 1],
        [0,     height, 1]
    ])
    """
    Scale by 3 to extend forward by 3 meters instead.
    """
    points_in_pixel_space = points_in_pixel_space * 3
    """
    To convert these "pixel space" points into "camera frame", we recall
    that the intrinsic matrix is what took us from "camera frame" to "pixel
    space". So, we just invert the intrinsic matrix.
    """
    intrinsic_matrix_inverse = np.linalg.inv(intrinsic)
    points_in_camera_frame = np.matmul(intrinsic_matrix_inverse, points_in_pixel_space.T).T
    """
    We now have [x y z] points. To use the extrinsic matrix to go from camera_frame -> rear_axle_middle_ground frame,
    we must add an extra dimension at the end to make the points [x y z 1]. This extra dimension is necessary simply
    because we make a translation, and the 1 is used to create coefficients (recall tx, ty, tz below).
    """
    points_in_camera_frame = np.concatenate((points_in_camera_frame, np.ones((4, 1))), axis=1)
    points_in_world_frame = np.matmul(invert_extrinsics(extrinsic), points_in_camera_frame.T).T
    """
    Plot the four points. Make connectors from the camera's ionposition towards the corners of the camera's FOV, to create
    a cone-like visualization.
    """
    for i in range(4):
        x, y, z, _ = points_in_world_frame[i] / points_in_world_frame[i, 3]
        ax.plot([tx, x], [ty, y], [tz, z], color='black')
    """
    Connect the corners.
    """
    x = [*points_in_world_frame[:, 0], points_in_world_frame[0, 0]]
    y = [*points_in_world_frame[:, 1], points_in_world_frame[0, 1]]
    z = [*points_in_world_frame[:, 2], points_in_world_frame[0, 2]]
    ax.plot(x, y, z, color='black')

def create_camera_extrinsics(x, y, z, yaw, is_camera=True):
    """
    An extrinsic camera matrix follows the following format:
        [R t]
    where R is a 3x3 rotation matrix and t is a 3x1 translation vector.
    """

    """
    The extrinsic matrix is something that takes a point in rear_axle_middle_ground frame
    and converts it to "camera" frame. The main difference between these (besides the angle/perspective of the camera)
    is the difference between the axes.

    For the car, "forward" is X, "right" is Y, and "up" is Z.
    For the camera, though, "forward" is Z, "right" is X, and "up" is Y (or -Y, if you go top->bottom in images).
    """
    
    """
    The first step is to translate. The input point is a vector [x, y, z, 1]. The last dimension is always 1. This is to help
    with scaling.

    Here is an explanation:
    [1 0 0 -tx][x]   [x - tx]
    [0 1 0 -ty][y] = [y - ty]
    [0 0 1 -tz][z]   [z - tz]
    [0 0 0   1][1]   [1     ]

    Consider this from the perspective of the camera. A point at (tx, ty, tz) must go to the point (0, 0, 0) from the perspective of the camera.
    So this is the first part.
    """
    if is_camera:
        translation = np.array([
            [1, 0, 0, -x],
            [0, 1, 0, -y],
            [0, 0, 1, -z],
            [0, 0, 0, 1],
        ])
    else:
        translation = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ])
    """
    The next step is to rotate. If we ignore the "yaw" component for a second, imagine the camera is pointing straight ahead.

    For a rotation matrix (which is orthonormal), every row represents what that part of the matrix is "detecting". To explain
    more clearly, consider this matrix:

    [0 1 0 0][x]   [y]
    [0 0 1 0][y] = [z]
    [1 0 0 0][z]   [x]
    [0 0 0 1][1]   [1]

    Recall:
    For the car, "forward" is X, "right" is Y, and "up" is Z.
    For the camera, though, "forward" is Z, "right" is X, and "up" is Y (or -Y, if you go top->bottom in images).

    This means that for the output of this matrix,
    Row 1 (the "x" row) is detecting Y values, by dot-producting [0 1 0 0] with [x y z 1]
    Row 2 (the "y" row) is detecting Z values, by dot-producting [0 0 1 0] with [x y z 1]
    Row 3 (the "z" row) is detecting Z values, by dot-producting [1 0 0 0] with [x y z 1]

    This makes sense if you consider a matrix multiplication onto a vector as dotting each row of the matrix with the vector
    to produce a new vector.

    Another thing this means is that if we invert the rotation matrix (which, by virtue of being orthonormal, amounts to a
    matrix transpose [making the rows -> columns and vice versa]), then multiplying by a point in "camera" coordinates outputs
    how the point would look in rear_axle_middle_ground coordinates.
    """
    rot = np.array([
        [-np.sin(yaw), np.cos(yaw), 0, 0],
        [0, 0, 1, 0],
        [np.cos(yaw), np.sin(yaw), 0, 0],
        [0, 0, 0, 1],
    ])
    Rt = rot @ translation
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
    result = np.concatenate((R_inv, -R_inv_C), -1)
    result = np.concatenate((result, np.zeros((1, 4))), 0)
    result[3, :3] = 0
    result[3, 3] = 1
    return result

def get_extrinsics(use_yaml=False):
    # Load extrinsics
    if use_yaml:
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
            'front_left': create_camera_extrinsics(2.184, 0.171, 0.422, -0.9599310886),
            'front_right': create_camera_extrinsics(2.184, -0.171, 0.422, +0.9599310886),
            'rear_left': create_camera_extrinsics(1.473, 0.140, 0.543, -2.2689280276),
            'rear_right': create_camera_extrinsics(1.473, -0.140, 0.543, +2.2689280276),
            'front_left_center': create_camera_extrinsics(2.235, 0.121, 0.422, 0),
            'front_right_center': create_camera_extrinsics(2.235, -0.121, 0.422, 0),
        }
    
    return extrinsics

def main():
    """
    Demo. Display the camera locations and their fields of view.
    """

    """ Load intrinsic and extrinsic matrices. """
    with open("calibration/intrinsic_matrices.json", "r") as f:
        intrinsics = json.load(f)
        intrinsics = {k: np.array(v) for k, v in intrinsics.items()}

    extrinsics = get_extrinsics()

    # Create a 3D plot
    fig = plt.figure()
    num_columns = 4
    axes: Axes3D = fig.add_subplot(1, num_columns, 1, projection='3d') # type: ignore

    # Plot each camera
    for camera in extrinsics.keys():
        print("Plotting", camera)
        plot_camera(axes, extrinsics[camera], intrinsics.get(camera, None))

    # [old] luminar extrinsic ideas
    luminar_front_extrinsics = create_camera_extrinsics(2.342, 0, 0.448, 0, is_camera=False)
    luminar_left_extrinsics = create_camera_extrinsics(1.549, 0.267, 0.543, 2.0943951024, is_camera=False)
    luminar_right_extrinsics = create_camera_extrinsics(1.549, -0.267, 0.543, -2.0943951024, is_camera=False)

    index = -1
    # For plotting LiDAR points
    if index > -1:
        with open(f"front_{index}.pkl", "rb") as f:
            xyz_front = pickle.load(f)[0]
            xyz_front = (luminar_front_extrinsics @ np.concatenate((xyz_front, np.ones((len(xyz_front), 1))), axis=1).T).T

        with open(f"left_{index}.pkl", "rb") as f:
            xyz_left = pickle.load(f)[0]
            xyz_left = (luminar_left_extrinsics @ np.concatenate((xyz_left, np.ones((len(xyz_left), 1))), axis=1).T).T

        with open(f"right_{index}.pkl", "rb") as f:
            xyz_right = pickle.load(f)[0]
            xyz_right = (luminar_right_extrinsics @ np.concatenate((xyz_right, np.ones((len(xyz_right), 1))), axis=1).T).T

        xyz = np.concatenate((xyz_front, xyz_left, xyz_right), axis=0)
        xyz = xyz[:, :3]
        # their luminar points are uncalibrated
        # xyz = xyz_front
        # xyz = xyz_left
        xyz = xyz[::10]
        xyz1 = np.concatenate((xyz, np.ones((len(xyz), 1))), axis=1)
        xyz = (extrinsics['front_right'] @ xyz1.T).T[:, :3]
        axes.scatter(*xyz.T, s=1, zdir='x') # type: ignore

    # Set axis labels
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')

    # Set plot limits
    size = 10
    axes.set_xlim(-size, size)
    axes.set_ylim(-size, size)
    axes.set_zlim(-size, size)

    display_points = np.array([
        (10, -3, 1),
        (10, 0, 1),
        (10, 3, 1),
    ])

    for color, point in zip(AXIS_COLORS, display_points):
        axes.scatter([point[0]], [point[1]], [point[2]], c=[color])

    """ plot renders. """
    for i, render_camera in enumerate(['front_left', 'front_left_center', 'front_right']):
        axes = fig.add_subplot(1, num_columns, 2 + i)

        camera = render_camera
        print(intrinsics[camera])

        intrinsics_expanded = np.ones((4, 4))
        intrinsics_expanded[:3, :3] = intrinsics[camera]

        camera_matrix = intrinsics_expanded @ extrinsics[camera]

        display_points_augmented = np.concatenate((display_points, np.ones((display_points.shape[0], 1))), axis=1)

        projected_points = (camera_matrix @ np.array(display_points_augmented).T).T
        """
        We started from [x y z 1] and got [pixel_x, pixel_y]. We must normalize away the remaining 2 coordinates.
        """
        projected_points = (projected_points / projected_points[:, [3]])[:, :3]
        projected_points = (projected_points / projected_points[:, [2]])[:, :2]
        axes.set_title("Camera " + camera)
        axes.scatter(projected_points.T[0], projected_points.T[1], c=AXIS_COLORS)
        axes.set_xlim(0, 2064)
        axes.set_ylim(0, 960)
        axes.set_aspect('equal')
        print(projected_points)

    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
