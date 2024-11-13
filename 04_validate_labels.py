# Will store the dataset in COCO format.
# Using a more powerful labeler with a bit of manual feedback.

import json
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


def cxcywh_to_x1y1x2y2(boxes: torch.Tensor):
    # (center x, center y, width, height) -> (x1, y1, x2, y2)
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes


def render_labels(image: np.ndarray, boxes: torch.Tensor):
    h, w, _ = image.shape

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


def has_overlapping_pair(boxes, iou_threshold=0.5):
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if calculate_iou(boxes[i], boxes[j]) > iou_threshold:
                return True

    return False


def no_track(reason: str, boxes: torch.Tensor, frame_id: int):
    return {
        "tracked": False,
        "reason": reason,
        "box_rejections": boxes,
        "frame_id": frame_id,
    }


def track(box, reason, frame_id):
    return {"tracked": True, "reason": reason, "box": box, "frame_id": frame_id}


def x1y1x2y2_to_centered(box: torch.Tensor):
    return torch.stack(
        [(box[..., 0] + box[..., 2]) / 2, (box[..., 1] + box[..., 3]) / 2], dim=-1
    )


def get_judgement_for_boxes(
    image, boxes, previous_detection, history, frame_id, distance_threshold
):
    """
    Returns (success, info).
    `info` contains the box, if detected, and the reason for the result.
    """

    if len(boxes) == 0:
        return no_track("no_detections", boxes, frame_id)

    if previous_detection is None:
        # Only give a detection to a human labeler if there is a single detection to check.
        if len(boxes) != 1:
            return no_track("ambiguous_boxes_for_human_label", boxes, frame_id)

        box = boxes[0]
        # Reject if the previous was a human_rejection or a human_rejection_track and we are close to that.
        for info in history[-2:]:
            if info["tracked"]:
                continue

            if info["reason"] in ["human_rejection", "human_rejection_track"]:
                # Check distance to that box.
                rejection = info["box_rejections"][0]
                distance = (
                    x1y1x2y2_to_centered(rejection) - x1y1x2y2_to_centered(box)
                ).norm()
                if distance < distance_threshold:
                    return no_track("human_rejection_track", boxes, frame_id)

        render_labels(image, boxes)
        if "y" == input("Good? (y/n)"):
            return track(boxes[0], "human_acceptance", frame_id)
        else:
            return no_track("human_rejection", boxes, frame_id)

    previous_center = x1y1x2y2_to_centered(previous_detection)
    box_centers = x1y1x2y2_to_centered(boxes)
    distances = (box_centers - previous_center).norm(dim=-1)
    # Sort boxes in order of distance.
    best_box_indexes = torch.argsort(distances)
    best_box = boxes[best_box_indexes[0]]
    best_box_distance = distances[best_box_indexes[0]]

    if best_box_distance > distance_threshold:
        return no_track("distance_rejection", boxes, frame_id)

    # Check for multiple overlapping boxes.
    if has_overlapping_pair(boxes[distances <= distance_threshold]):
        return no_track("overlapping_boxes_rejection", boxes, frame_id)

    return track(best_box, "distance_acceptance", frame_id)


def remove_boxes_containing_points(
    boxes: torch.Tensor, X: torch.Tensor, Y: torch.Tensor
):
    # Remove ``boxes" that cover the entire screen, as well as boxes
    # with a terrible "aspect ratio".
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    boxes = boxes[((widths * heights) < 0.9) & (heights < widths)]

    # Remove boxes containing forbidden points.
    boxes_unsqueezed = boxes.unsqueeze(-1).expand(-1, -1, len(X))
    contains_forbidden_x = (boxes_unsqueezed[:, 0] < X) & (X < boxes_unsqueezed[:, 2])
    contains_forbidden_y = (boxes_unsqueezed[:, 1] < Y) & (Y < boxes_unsqueezed[:, 3])
    contains_forbidden = torch.any(contains_forbidden_x & contains_forbidden_y, dim=-1)
    boxes = boxes[~contains_forbidden]

    return boxes


IMAGE_SIZE = (2064, 960)
FORBIDDEN_POINTS = {
    "front_left_center": [(2000, 900)],
    "rear_right": [(2000, 900), (1500, 700)],
}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    IMAGES_FOLDER = "images/M-MULTI-SLOW-KAIST/rear_right"
    RAW_LABELS_FOLDER = "labels/rear_right/raw_detection_results"

    # Define a set of points that are on *our* car.
    # If the bounding box contains these points, we
    # filter this box out as "not an opponent".
    # Stores as parallel tensors.
    forbidden_points = [(2000, 900)]
    forbidden_x = torch.tensor([x for (x, _) in forbidden_points]) / IMAGE_SIZE[0]
    forbidden_y = torch.tensor([y for (_, y) in forbidden_points]) / IMAGE_SIZE[1]

    torch.random.manual_seed(42)

    most_recent_good_box = None
    confidence_distance = 50
    lookback_distance = 5
    steps_since_good_detection = 0

    plt.figure(figsize=(8, 6))

    with open("log_with_frame_ids.pkl", "rb") as f:
        existing_history = pickle.load(f)
        existing_history = {item["frame_id"]: item for item in existing_history}
    n_images = len(os.listdir(IMAGES_FOLDER))
    history = []

    try:
        for image_number in range(1, n_images + 1):
            image_path = os.path.join(IMAGES_FOLDER, f"rear_right_{image_number}.jpeg")
            label_path = os.path.join(
                RAW_LABELS_FOLDER, f"rear_right_{image_number}.json"
            )
            if not (os.path.exists(image_path) and os.path.exists(label_path)):
                continue

            frame = np.array(PIL.Image.open(image_path))
            with open(label_path) as f:
                label = json.load(f)

            if len(label["boxes"]) == 0:
                continue

            boxes = torch.tensor(label["boxes"])
            logits = torch.tensor(label["logits"])

            logging.info("frame %d.", image_number)

            print("Original boxes:", boxes)

            boxes = cxcywh_to_x1y1x2y2(boxes)
            boxes = remove_boxes_containing_points(boxes, forbidden_x, forbidden_y)

            result = get_judgement_for_boxes(
                frame,
                boxes,
                most_recent_good_box,
                history,
                image_number + 1,
                confidence_distance,
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
    except KeyboardInterrupt:
        logging.info("Interrupted.")

    with open("log_complete.pkl", "wb") as f:
        pickle.dump(history, f)


if __name__ == "__main__":
    main()
