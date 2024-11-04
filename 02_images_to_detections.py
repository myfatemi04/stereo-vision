"""
Usage: python detect_cars.py </path/to/images> <output_file>
"""

import json
import os

import click

# Grounding DINO
import groundingdino.datasets.transforms as T
import PIL.Image
import torch
import tqdm
from groundingdino.models import build_model
from groundingdino.models.GroundingDINO.groundingdino import GroundingDINO
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download

CHECKPOINT_REPO_ID = "ShilongLiu/GroundingDINO"
CHECKPOINT_FILENAME = "groundingdino_swinb_cogcoor.pth"
CHECKPOINT_CONFIG_FILENAME = "GroundingDINO_SwinB.cfg.py"
INFERENCE_TRANSFORM = T.Compose(
    [
        # Selects a "random" size from the list [800,]
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def load_model_hf(repo_id, filename, ckpt_config_filename, device) -> GroundingDINO:
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


# Infer on image folder
@click.command()
@click.option("--input_folder", "-i", required=True, type=str, help="Path to images.")
@click.option(
    "--output_folder",
    "-o",
    required=True,
    type=str,
    help="Folder to store resulting images in.",
)
@click.option(
    "--text_prompt",
    "-p",
    default="racecar",
    type=str,
    help="Text prompt to use for grounding.",
)
@click.option(
    "--box_threshold",
    default=0.3,
    type=float,
    help="Bounding box confidence threshold for filtering detections.",
)
@click.option(
    "--text_threshold",
    default=0.25,
    type=float,
    help="Text confidence threshold for filtering detections.",
)
def detect_with_foundation_model(
    input_folder: str,
    output_folder: str,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model_hf(
        CHECKPOINT_REPO_ID,
        CHECKPOINT_FILENAME,
        CHECKPOINT_CONFIG_FILENAME,
        device,
    )
    model = model.to(device)

    os.makedirs(f"{output_folder}/yolo/labels")
    os.makedirs(f"{output_folder}/raw_detection_results")

    for filename in tqdm.tqdm(
        sorted(os.listdir(input_folder)),
        desc="Running inference with foundation model.",
    ):
        if not filename.endswith(".png") and not filename.endswith(".jpg"):
            continue

        image_id = filename.split(".")[0]
        image_path = os.path.join(input_folder, filename)
        raw_image = PIL.Image.open(image_path).convert("RGB")
        image, _ = INFERENCE_TRANSFORM(raw_image, None)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )

        with open(f"{output_folder}/yolo/labels/{image_id}.txt", "w") as f:
            for box in boxes:
                center_x, center_y, width, height = box
                f.write(f"0 {center_x} {center_y} {width} {height}\n")

        with open(f"{output_folder}/raw_detection_results/{image_id}.json", "w") as f:
            json.dump(
                {
                    "boxes": boxes.tolist(),
                    "logits": logits.tolist(),
                    "phrases": phrases,
                },
                f,
            )


if __name__ == "__main__":
    detect_with_foundation_model()
