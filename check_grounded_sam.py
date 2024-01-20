# Will store the dataset in COCO format.
# Using a more powerful labeler with a bit of manual feedback.

import logging
import torch
import cv2

def convert_boxes(boxes: torch.Tensor):
    # (center x, center y, width, height) -> (x1, y1, x2, y2)
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

src = 'M-MULTI-SLOW-KAIST/front_left_center_results.pt'
results = torch.load(src)
logging.info("Loaded results from %s", src)

cap = cv2.VideoCapture("M-MULTI-SLOW-KAIST/front_left_center.mp4")

# Define a set of points that are on *our* car.
# If the bounding box contains these points, we
# filter this box out as "not an opponent".
# Stores as paralle tensors.
forbidden_points = [(2000, 900)]
forbidden_x = torch.tensor([x for (x, _) in forbidden_points]) / 2064
forbidden_y = torch.tensor([y for (_, y) in forbidden_points]) / 960

skip_empty = False

torch.random.manual_seed(42)
num_tests = 100
selection = {x.item() for x in torch.randperm(len(results))[:num_tests]}

annotation_indexes = []
annotations = []

frames_with_cars = 0

for i, (boxes, confidences, labels) in enumerate(results):
    _, frame = cap.read()
    h, w, _ = frame.shape

    if i not in selection:
        continue

    logging.info("Frame %d.", i)

    scale = torch.tensor([w, h, w, h])
    # Boxes use proportional coordinates
    boxes = convert_boxes(boxes)

    # Remove ``boxes" that cover the entire screen.
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    boxes = boxes[areas < 0.9]

    # Remove boxes containing forbidden points.
    contains_forbidden_x = (boxes[:, 0] < forbidden_x) & (forbidden_x < boxes[:, 2])
    contains_forbidden_y = (boxes[:, 1] < forbidden_y) & (forbidden_y < boxes[:, 3])
    contains_forbidden = contains_forbidden_x & contains_forbidden_y
    boxes = boxes[~contains_forbidden]

    # plot boxes
    if len(boxes) > 0:
        for (x1, y1, x2, y2) in boxes * scale:
            logging.debug("Box location: (%.2f, %.2f, %.2f, %.2f)", x1.item(), y1.item(), x2.item(), y2.item())
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        cv2.imwrite("frame.png", frame)
        frames_with_cars += 1
    else:
        logging.info("No cars detected.")

        if skip_empty:
            continue
        else:
            cv2.imwrite("frame.png", frame)

    # Exclude boxes that contain our car's wheels.
    annotation_indexes.append(i)
    annotations.append(input("Result [fp/tp/fn/tn; if multiple, separate by space]: "))

print(annotation_indexes)
print(annotations)

# Note: Frame 1470 should be fp.
# Frame 2126 has a bad bounding box.
# Frame 2542 is a sort of edge case (only the wheel of the other car is showing, but do we want to detect these?)
# Frame 4280 overlapping bounding boxes. why weren't these non max-suppressed?

"""
[25, 131, 156, 179, 195, 212, 284, 346, 384, 491, 582, 675, 679, 836, 863, 876, 927, 961, 1019, 1085, 1131, 1142, 1148, 1196, 1244, 1388, 1470, 1498, 1635, 1695, 1730, 1731, 1807, 1896, 2047, 2099, 2126, 2188, 2348, 2373, 2452, 2458, 2542, 2559, 2747, 2752, 2833, 2837, 2841, 2871, 2916, 2960, 2975, 3060, 3181, 3195, 3208, 3221, 3396, 3422, 3454, 3588, 3649, 3658, 3659, 3662, 3678, 3756, 3787, 3858, 3869, 3877, 3896, 3918, 3921, 3922, 3957, 4059, 4061, 4280, 4337, 4346, 4404, 4422, 4452, 4542, 4543, 4573, 4605, 4614, 4620, 4654, 4665, 4705, 4743, 4769, 4795, 4809, 4943, 5048]
['tn', 'fn', 'tp', 'tp', 'tp', 'tp', 'tp', 'tp', 'tp', 'tp', 'tp fp', 'tp fp', 'tp fp', 'tp', 'tp', 'tp', 'tp', 'tp', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'fp', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'tp', 'tp', 'tp', 'tp', 'tp fp', 'tp fp fp', 'tp', 'tp', '--', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'fp', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'tp', 'tn', 'tp', 'tp', 'tp', 'tp', 'tp', 'fp', 'fp', 'tn', 'tn', 'tn', 'fn', 'tp', 'tp', 'tp', 'tp', 'tp', 'tp', '--', 'tp fp', 'tp fp', 'tp fp', 'tp', 'tp', 'tp', 'tp', 'tp', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn', 'tn']
"""
