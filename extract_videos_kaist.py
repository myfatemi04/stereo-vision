from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import cv2
import numpy as np
import json
import open3d
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_pointcloud2(parsed_message):
    points = np.frombuffer(parsed_message.data, np.float32).reshape((parsed_message.height, parsed_message.width, 6))
    point_xyzs = points[:, :, :3].reshape((-1, 3))
    point_intensities = points[:, :, 3].reshape((-1))
    point_rings = points[:, :, 4].reshape((-1))
    point_times = points[:, :, 5].reshape((-1))
    return (point_xyzs, point_intensities, point_rings, point_times)

def main():
    # Dictionary to store VideoWriter instances for each camera
    video_writers = {}
    timestamps = {}
    count = 0

    # Create reader instance and open for reading
    with Reader('./M-MULTI-SLOW-KAIST') as reader:
        logging.info("Opened file.")
        # Topic names: luminar_{front, left, right}_points
        # Wait until we have all three to combine
        previous_lidar_messages = {}
        # Iterate over messages
        for connection, timestamp, rawdata in reader.messages():
            if connection.msgtype == 'sensor_msgs/msg/PointCloud2':
                parsed_message = deserialize_cdr(rawdata, connection.msgtype)
                previous_lidar_messages[connection.topic] = parsed_message

                if len(previous_lidar_messages) == 3:
                    # Note: Must transform all LiDARs to have the same 
                    xys = np.concatenate([parse_pointcloud2(previous_lidar_messages['/vehicle_8/luminar_front_points'])[0],
                                          parse_pointcloud2(previous_lidar_messages['/vehicle_8/luminar_left_points'])[0],
                                          parse_pointcloud2(previous_lidar_messages['/vehicle_8/luminar_right_points'])[0]])
                    cloud = open3d.geometry.PointCloud()
                    cloud.points = open3d.utility.Vector3dVector(xys)
                    open3d.io.write_point_cloud(f'lidar_{count}.ply', cloud)
                    count += 1
                    previous_lidar_messages.clear()
                    logging.info("Filled all three messages.")
                    exit()

            elif connection.msgtype == 'sensor_msgs/msg/CompressedImage':
                continue

                topic = connection.topic

                # Check if the message topic matches the specified format
                if topic.startswith('/vehicle_8/camera/') and topic.endswith('/image/compressed'):
                    camera_id = topic.split('/')[3]
                    count += 1
                    print("Frames processed:", count, end='\r')
                    if camera_id not in timestamps:
                        print("Adding", camera_id)
                        # Define the codec and create a VideoWriter instance for the camera
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs if needed
                        out_file = f'{camera_id}.mp4'
                        width, height = (2064, 960)
                        video_writers[camera_id] = cv2.VideoWriter(out_file, fourcc, 20, (width, height))
                        timestamps[camera_id] = []

                    msg = deserialize_cdr(rawdata, connection.msgtype)
                    np_arr = np.fromstring(msg.data, np.uint8)
                    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    video_writers[camera_id].write(image_np)

                    # Write the image frame to the corresponding video file
                    timestamps[camera_id].append(timestamp)

    # Release all VideoWriter instances
    for camera_id, writer in video_writers.items():
        writer.release()

    with open("timestamps.json", "w") as f:
        json.dump(timestamps, f)

if __name__ == '__main__':
    main()
