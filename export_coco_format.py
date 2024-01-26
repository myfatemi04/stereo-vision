# Will store the dataset in COCO format.
# Using a more powerful labeler with a bit of manual feedback.

import logging
import torch
import cv2
import os
import json

import tqdm

def convert_boxes(boxes: torch.Tensor):
    # (center x, center y, width, height) -> (x1, y1, x2, y2)
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes

def main(bag_id, camera_name, output_dir):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "images"))

    foundation_model_annotations_path = f'{bag_id}/{camera_name}_results.pt'
    foundation_model_annotations = torch.load(foundation_model_annotations_path)

    logging.info("Loaded results from %s", foundation_model_annotations_path)

    cap = cv2.VideoCapture(f"{bag_id}/{camera_name}.mp4")

    # Define a set of points that are on *our* car.
    # If the bounding box contains these points, we
    # filter this box out as "not an opponent".
    # Stores as paralle tensors.
    forbidden_points = [(2000, 900)]
    forbidden_x = torch.tensor([x for (x, _) in forbidden_points]) / 2064
    forbidden_y = torch.tensor([y for (_, y) in forbidden_points]) / 960

    torch.random.manual_seed(42)

    # Assuming you have the necessary COCO annotation structure
    coco_metadata = {
        "info": {"description": "Your description", "version": "1.0", "year": 2024, "contributor": "Michael Fatemi"},
        "licenses": [{"id": 1, "name": "Your License", "url": "Your License URL"}],
        "categories": [{"id": 1, "name": "racecar", "supercategory": "vehicle"}],  # Add more categories if needed
        "images": [],
        "annotations": [],
    }

    for image_id, (boxes, confidences, labels) in tqdm.tqdm(enumerate(foundation_model_annotations), desc='Saving images and annotations', total=len(foundation_model_annotations)):
        _, frame = cap.read()
        h, w, _ = frame.shape

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

        # Assuming you have the image size information
        # Use consistent image_id.
        # Only write images to disk if they do not exist there already.
        image_filename = "image_%d.png" % image_id
        image_path = os.path.join(output_dir, "images", image_filename)
        if not os.path.exists(image_path):
            cv2.imwrite(image_path, frame)

        image_info = {"id": image_id, "file_name": image_filename, "width": 2064, "height": 960}  # Update width and height
        coco_metadata["images"].append(image_info)

        # Rescale boxes to original image size.
        for box_counter, box in enumerate(boxes * scale):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)

            annotation_id = image_id * 1000 + box_counter
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # Assuming only one category (car)
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "area": float(area),
                "iscrowd": 0
            }

            coco_metadata["annotations"].append(annotation)
            box_counter += 1
    
    # Write results.
    output_path = os.path.join(output_dir, "annotations.json")
    with open(output_path, "w") as json_file:
        json.dump(coco_metadata, json_file, indent=4)

if __name__ == '__main__':
    BAG_ID = 'M-MULTI-SLOW-KAIST'
    CAMERA_NAME = 'front_left_center'
    output_dir = "M-MULTI-SLOW-KAIST_coco_format"
    main(BAG_ID, CAMERA_NAME, output_dir)
