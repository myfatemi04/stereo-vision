# Vision Model Training Pipelines

## Pipeline 1: Data Preprocessing

### Extracting images from bags

Use `01_bag_to_images.py` to extract images from bag files.
```
Usage: 01_bag_to_images.py [OPTIONS]

Options:
  --bag_dir TEXT        Path to the bag file.  [required]
  --out_dir TEXT        Path to store extracted content to.  [required]
  --include_camera      Whether to extract camera images from the bag file.
  --include_lidar       Whether to extract lidar point clouds from the bag
                        file.
  --overwrite_existing  Whether to overwrite existing content in the output
                        directory. If not, then the extraction will be skipped
                        for that file.
  --help                Show this message and exit.
```

## Pipeline 2: Data Labeling

### Creating foundation model labels for a folder of images

Use `02_images_to_detections.py` to extract detections for images, using the foundation model. This stores detection results in the following manner:
- `<output_folder>/yolo/labels/<image_filename>.txt`: Image labels in YOLO format.
- `<output_folder>/raw_detection_results/<image_filename>.json`: Contains a dictionary with the keys "boxes", "logits", and "phrases". 
  - "boxes" is a tensor of shape (N, 4) containing the bounding boxes of the detections. They are stored in the format `(x_center, y_center, width, height)`, normalized to image coordinates (i.e. ranging from $0$ to $1$).
  - "logits" is a tensor of shape (N,) containing the confidence scores of the detections.
  - "phrases" is a list of strings containing the grounded phrases for the detections (in this case, all are equal to the value of the `text_prompt` input parameter).
```
Usage: 02_images_to_detections.py [OPTIONS]

Options:
  -i, --input_folder TEXT   Path to images.  [required]
  -o, --output_folder TEXT  Folder to store resulting images in.  [required]
  -p, --text_prompt TEXT    Text prompt to use for grounding.
  --box_threshold FLOAT     Bounding box confidence threshold for filtering
                            detections.
  --text_threshold FLOAT    Text confidence threshold for filtering
                            detections.
  --help                    Show this message and exit.
```
