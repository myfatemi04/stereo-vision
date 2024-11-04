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

