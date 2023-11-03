from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
import cv2
import numpy as np
import json

# Dictionary to store VideoWriter instances for each camera
video_writers = {}
timestamps = {}
count = 0

# Create reader instance and open for reading
with Reader('./M-MULTI-SLOW-KAIST') as reader:
    # Iterate over messages
    for connection, timestamp, rawdata in reader.messages():
        if connection.msgtype == 'sensor_msgs/msg/CompressedImage':
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
