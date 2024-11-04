# Will store the dataset in COCO format.
# Using a more powerful labeler with a bit of manual feedback.

import logging
import os
import pickle

import matplotlib.patches as patches
import numpy as np
import PIL.Image
import torch
from matplotlib import pyplot as plt


# Finds the area of intersection between two boxes.
def calculate_intersection(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1
    box2_x1, box2_y1, box2_x2, box2_y2 = box2

    # Calculate the (x, y) coordinates of the intersection rectangle's corners
    x1 = max(box1_x1, box2_x1)
    y1 = max(box1_y1, box2_y1)
    x2 = min(box1_x2, box2_x2)
    y2 = min(box1_y2, box2_y2)

    # If there is no intersection, return 0
    if x1 > x2 or y1 > y2:
        return 0

    # Calculate the area of the intersection rectangle
    intersection_area = (x2 - x1) * (y2 - y1)

    return intersection_area


def calculate_area(box):
    return (box[3] - box[1]) * (box[2] - box[0])


def calculate_iou(box1, box2):
    total_area = calculate_area(box1) + calculate_area(box2)
    intersection = calculate_intersection(box1, box2)
    union = (
        total_area - intersection
    )  # total_area simply overcounts the intersection, so we can subtract the union
    return intersection / union


def convert_boxes(boxes: torch.Tensor):
    # (center x, center y, width, height) -> (x1, y1, x2, y2)
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes


def render_labels(image, boxes):
    h, w, _ = image.shape
    image = image[:, :, ::-1]

    plt.clf()
    plt.imshow(image)

    scale = torch.tensor([w, h, w, h])
    for x1, y1, x2, y2 in boxes * scale:
        x = x1.item()
        y = y1.item()
        w = (x2 - x1).item()
        h = (y2 - y1).item()
        logging.debug(
            "Box location: (%.2f, %.2f, %.2f, %.2f)",
            x1.item(),
            y1.item(),
            x2.item(),
            y2.item(),
        )
        # Create a Rectangle patch
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=4, edgecolor="g", facecolor="none"
        )

        # Add the patch to the Axes
        plt.gca().add_patch(rect)

    plt.pause(0.02)


def has_multiple_overlapping_boxes(boxes, iou_threshold=0.5):
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            box1 = boxes[i]
            box2 = boxes[j]
            if calculate_iou(box1, box2) > iou_threshold:
                return True

    return False


def no_track(reason, boxes):
    return {"tracked": False, "reason": reason, "box_rejections": boxes}


def track(box, reason):
    return {"tracked": True, "reason": reason, "box": box}


def center(box):
    return torch.stack(
        [(box[..., 0] + box[..., 2]) / 2, (box[..., 1] + box[..., 3]) / 2], dim=-1
    )


def get_judgement_for_boxes(
    image, boxes, previous_detection, history, distance_threshold=50
):
    """
    Returns (success, info).
    `info` contains the box, if detected, and the reason for the result.
    """

    if len(boxes) == 0:
        return no_track("no_detections", boxes)

    if previous_detection is None:
        # Only give a detection to a human labeler if there is a single detection to check.
        if len(boxes) != 1:
            return no_track("ambiguous_boxes_for_human_label", boxes)

        box = boxes[0]
        # Reject if the previous was a human_rejection or a human_rejection_track and we are close to that.
        for info in history[-2:]:
            if info["tracked"]:
                continue

            if info["reason"] in ["human_rejection", "human_rejection_track"]:
                # Check distance to that box.
                rejection = info["box_rejections"][0]
                distance = (center(rejection) - center(box)).norm()
                if distance < distance_threshold:
                    return no_track("human_rejection_track", boxes)

        render_labels(image, boxes)
        if "y" == input("Good? (y/n)"):
            return track(boxes[0], "human_acceptance")
        else:
            return no_track("human_rejection", boxes)

    previous_center = center(previous_detection)
    box_centers = center(boxes)
    distances = (box_centers - previous_center).norm(dim=-1)
    # Sort boxes in order of distance.
    best_box_indexes = torch.argsort(distances)
    best_box = boxes[best_box_indexes[0]]
    best_box_distance = distances[best_box_indexes[0]]

    if best_box_distance > distance_threshold:
        return no_track("distance_rejection", boxes)

    # Check for multiple overlapping boxes.
    if has_multiple_overlapping_boxes(boxes[distances <= distance_threshold]):
        return no_track("overlapping_boxes_rejection", boxes)

    return track(best_box, "distance_acceptance")


def preliminary_box_filtering(boxes, forbidden_x, forbidden_y):
    # Remove ``boxes" that cover the entire screen, as well as boxes
    # with a terrible "aspect ratio".
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    boxes = boxes[((widths * heights) < 0.9) & (heights < widths)]

    # Remove boxes containing forbidden points.
    boxes_unsqueezed = boxes.unsqueeze(-1).expand(-1, -1, len(forbidden_x))
    contains_forbidden_x = (boxes_unsqueezed[:, 0] < forbidden_x) & (
        forbidden_x < boxes_unsqueezed[:, 2]
    )
    contains_forbidden_y = (boxes_unsqueezed[:, 1] < forbidden_y) & (
        forbidden_y < boxes_unsqueezed[:, 3]
    )
    contains_forbidden = torch.any(contains_forbidden_x & contains_forbidden_y, dim=-1)
    boxes = boxes[~contains_forbidden]

    return boxes


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    CAMERA_NAME = "front_left_center"
    BAG_FOLDER = os.path.expanduser("~/bags/extracted/M-MULTI-SLOW-KAIST")
    IMAGE_FOLDER = os.path.join(BAG_FOLDER, "camera", CAMERA_NAME)
    LABELS_PATH = os.path.join(BAG_FOLDER, "camera-labels", CAMERA_NAME + "_results.pt")

    raw_labels = torch.load(LABELS_PATH)
    logging.info("Loaded results from %s", LABELS_PATH)

    # Define a set of points that are on *our* car.
    # If the bounding box contains these points, we
    # filter this box out as "not an opponent".
    # Stores as paralle tensors.
    forbidden_points = [(2000, 900)]
    forbidden_x = torch.tensor([x for (x, _) in forbidden_points]) / 2064
    forbidden_y = torch.tensor([y for (_, y) in forbidden_points]) / 960

    torch.random.manual_seed(42)

    most_recent_good_box = None
    CONFIDENCE_DISTANCE = 50
    lookback_distance = 5
    steps_since_good_detection = 0

    plt.figure(figsize=(8, 6))

    history = []

    try:
        for i, (boxes, confidences, labels) in enumerate(raw_labels):
            frame = PIL.Image.open(os.path.join(IMAGE_FOLDER, f"image_{i + 1}.png"))
            frame = np.array(frame)

            logging.info("frame %d.", i)

            print("Original boxes:", boxes)

            ### Preliminary filtering. ###
            # NOTE: Boxes use proportional coordinates
            boxes = convert_boxes(boxes)
            boxes = preliminary_box_filtering(boxes, forbidden_x, forbidden_y)

            result = get_judgement_for_boxes(
                frame, boxes, most_recent_good_box, history, CONFIDENCE_DISTANCE
            )
            print(result)

            # Invalidate if it's been too long.
            if not result["tracked"]:
                steps_since_good_detection += 1
                if steps_since_good_detection >= lookback_distance:
                    most_recent_good_box = None
            else:
                steps_since_good_detection = 0
                most_recent_good_box = result["box"]

            history.append(result)
            render_labels(frame, boxes)
    except:
        pass

    with open("log.pkl", "wb") as f:
        pickle.dump(history, f)


if __name__ == "__main__":
    main()
