"""
Usage: python detect_cars.py </path/to/images> <output_file>
"""

import glob
import pickle
import sys
from typing import Tuple

# Grounding DINO
import groundingdino.datasets.transforms as T
import PIL.Image
import torch
import tqdm
from groundingdino.models import build_model
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_hf(repo_id, filename, ckpt_config_filename, device):
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


transform = T.Compose(
    [
        # Selects a "random" size from the list [800,]
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

DetectionResult = Tuple[torch.Tensor, torch.Tensor, list]


# Infer on image folder
def infer_image_folder(input_path: str, output_file: str):
    # Use this command for evaluate the Grounding DINO model
    # Or you can download the model by yourself
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filename = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    model = load_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device)
    model = model.to(device)

    # Detection settings
    TEXT_PROMPT = "car"
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25

    image_paths = sorted(
        list(glob.glob(f"{input_path}/*.png")) + list(glob.glob(f"{input_path}/*.jpg"))
    )
    results = []

    for image_path in tqdm.tqdm(image_paths, desc="Detecting cars."):
        raw_image = PIL.Image.open(image_path).convert("RGB")
        image, _ = transform(raw_image, None)
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=device,
        )
        results.append((boxes, logits, phrases))

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print(f"Saved results to {output_file}.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python detect_cars.py </path/to/images> <output_file>")
        exit(1)

    input_path = sys.argv[1]
    output_file = sys.argv[2]
    infer_image_folder(input_path, output_file)
