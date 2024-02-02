from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

def main(bag_path):
    with Reader(bag_path) as reader:
        topics = [topic_name for (topic_name, topic) in reader.topics.items() if topic.msgtype == 'sensor_msgs/msg/CameraInfo']
        unseen_topics = set(topics)
        remaining = len(topics)
        matrices = {}
        distortions = {}
        print("Detected CameraInfo topics:", topics)
        for (connection, timestamp, raw_data) in reader.messages():
            topic = connection.topic
            msgtype = connection.msgtype

            if topic in unseen_topics:
                data = deserialize_cdr(raw_data, msgtype)

                # See https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
                intrinsics = list(data.k)
                row1 = intrinsics[0:3]
                row2 = intrinsics[3:6]
                row3 = intrinsics[6:9]
                # Manually override cx and cy values
                # row1[-1] = 2064/2
                # row2[-1] = 960/2
                camera_name = topic.split('/')[-2]
                matrices[camera_name] = [row1, row2, row3]
                distortions[camera_name] = list(data.d)
                unseen_topics.remove(topic)
                remaining -= 1

                if remaining == 0:
                    break

        print("Intrinsic matrices:")
        print(matrices)
        print("Distortion coefficients:")
        print(distortions)

        # Output as JSON
        import json

        with open('intrinsic_models.json', 'w') as f:
            json.dump({"matrices": matrices, "distortions": distortions}, f, indent=2)

if __name__ == '__main__':
    import sys

    # bag_path = 'M-MULTI-SLOW-KAIST'

    if len(sys.argv) != 2:
        print("Usage: python get_camera_matrices_from_bag.py <bag_path>")
        exit(1)

    bag_path = sys.argv[1]
    main(bag_path)
