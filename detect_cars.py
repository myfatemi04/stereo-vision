

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict

from huggingface_hub import hf_hub_download

import torch
import PIL.Image
import numpy as np
import dataclasses

device = 'cuda'

def load_model_hf(repo_id, filename, ckpt_config_filename, device):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

transform = T.Compose([
    # Selects a "random" size from the list [800,]
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

@dataclasses.dataclass
class DetectionResult:
    boxes: torch.Tensor
    logits: torch.Tensor
    phrases: list

def detect_cars() -> DetectionResult:
    pass

def main():
    # Use this command for evaluate the Grounding DINO model
    # Or you can download the model by yourself
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)

    TEXT_PROMPT = "car"
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25

    raw_image = PIL.Image.open("raw_image.jpg").convert("RGB")
    image, _ = transform(raw_image, None)
    image_source = np.array(raw_image)

    # tensor, tensor, list
    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD,
        device=device,
    )

    print(type(boxes), type(logits), type(phrases))
    print(boxes.shape, logits.shape, phrases.shape)

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB

    # Save demo image
    PIL.Image.fromarray(annotated_frame).save("detection.png")

main()
