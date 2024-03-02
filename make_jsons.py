from matrices import get_camera_extrinsics
import json

ext = get_camera_extrinsics()

with open("calibration/intrinsic_models.json") as f:
    int_dst = json.load(f)

camera_info = {
    'front_left': {},
    'front_left_center': {},
    'front_right': {},
    # 'front_right_center': {},
}

for key in ext:
    if key == 'front_right_center':
        continue
    if key not in camera_info:
        camera_info[key] = {}

    camera_info[key]['ext'] = list(list(x) for x in ext[key])
    camera_info[key]['int'] = list(list(x) for x in int_dst['matrices'][key])
    camera_info[key]['dis'] = list(int_dst['distortions'][key])

with open("cameras.json", "w") as f:
    json.dump(camera_info, f)
