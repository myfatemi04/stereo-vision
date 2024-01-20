"""
Usage: python detect_cars.py <video_path>
"""

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict, preprocess_caption
import bisect

from huggingface_hub import hf_hub_download

import torch
import time
import PIL.Image
import numpy as np
from typing import List, Tuple
import tqdm
import cv2

device = 'cuda:0'

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

DetectionResult = Tuple[torch.Tensor, torch.Tensor, list]

# @dataclasses.dataclass
# class DetectionResult:
#     boxes: torch.Tensor
#     logits: torch.Tensor
#     phrases: list

def detect_from_video(
    model,
    video_path: str,
    caption="car",
    box_threshold=0.3,
    text_threshold=0.25,
    frame_batch_size=32
) -> List[DetectionResult]:
    cap = cv2.VideoCapture(video_path)

    start_time = time.time()
    results: List[DetectionResult] = []
    nbatches = 0

    def consume_batch(batch):
        nonlocal nbatches

        nbatches += 1
        print("Processing batch", nbatches, f"t={(time.time()-start_time)/nbatches:.2f}s/batch", end='\r')

        for i in range(len(batch)):
            image = batch[i][0]
            boxes, logits, phrases = predict(
                model=model, 
                image=image,
                caption=caption, 
                box_threshold=box_threshold, 
                text_threshold=text_threshold,
                device=device,
            )
            results.append((boxes, logits, phrases))

        # image_tensor = torch.stack([batch[i][0] for i in range(len(batch))])
        # box_results, logit_results, phrase_results = predict_batch(model, image_tensor, caption, box_threshold, text_threshold)
        # for boxes, logits, phrases in zip(box_results, logit_results, phrase_results):
        #     results.append(DetectionResult(boxes, logits, phrases))

    batch = []
    while cap:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to desired format
        raw_image = PIL.Image.fromarray(frame, mode="RGB")
        image, _ = transform(raw_image, None)
        image_source = frame

        batch.append((image.to(device), image_source))

        if len(batch) == frame_batch_size:
            consume_batch(batch)
            batch.clear()

    if len(batch) > 0:
        consume_batch(batch)
        batch.clear()

    return results

# Adapted from GroundingDINO/groundingdino/utils/inference.py:predict
def predict_batch(
    model,
    image_batch: torch.Tensor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    remove_combined: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[List[str]]]:
    caption = preprocess_caption(caption=caption)

    with torch.no_grad():
        outputs = model(image_batch, captions=[caption])
        # outputs = model(image[None], captions=[caption])

    # prediction_logits.shape = (nq, 256)
    Prediction_logits = outputs["pred_logits"].cpu().sigmoid()
    # prediction_boxes.shape = (nq, 4)
    Prediction_boxes = outputs["pred_boxes"].cpu()

    box_results = []
    logit_results = []
    phrase_results = []
    
    for (prediction_logits, prediction_boxes) in zip(Prediction_logits, Prediction_boxes):
        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)
        
        if remove_combined:
            sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
            
            phrases = []
            for logit in logits:
                max_idx = logit.argmax()
                insert_idx = bisect.bisect_left(sep_idx, max_idx)
                right_idx = sep_idx[insert_idx]
                left_idx = sep_idx[insert_idx - 1]
                phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
        else:
            phrases = [
                get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
                for logit
                in logits
            ]
        
        box_results.append(boxes)
        logit_results.append(logits.max(dim=1)[0])
        phrase_results.append(phrases)

    return box_results, logit_results, phrase_results

def demo():
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
    print(boxes.shape, logits.shape)

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB

    # Save demo image
    PIL.Image.fromarray(annotated_frame).save("detection.png")

def render_annotation_video(input_filename, annotations, output_filename):
    cap = cv2.VideoCapture(input_filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs if needed
    width, height = (2064, 960)
    writer = cv2.VideoWriter(output_filename, fourcc, 20, (width, height))

    for i in tqdm.tqdm(range(len(annotations)), desc='Generating annotated video'):
        ret, frame = cap.read()
        boxes, logits, phrases = annotations[i]
        annotated_frame = annotate(image_source=frame, boxes=boxes, logits=logits, phrases=phrases)
        writer.write(annotated_frame)
        
    writer.release()

def main(input_path):
    # Use this command for evaluate the Grounding DINO model
    # Or you can download the model by yourself
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
    ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

    model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)
    model = model.to(device)
    results = detect_from_video(model, video_path=input_path, frame_batch_size=1)
    torch.save(results, input_path.replace(".mp4", "_results.pt"))

# Video annotator
# input_filename = "M-MULTI-SLOW-KAIST/front_left.mp4"
# output_filename = "M-MULTI-SLOW-KAIST/front_left_annotated.mp4"
# annotations = torch.load("M-MULTI-SLOW-KAIST/front_left_results.pt")

# render_annotation_video(
#     input_filename,
#     annotations,
#     output_filename,
# )

for input_path in [
    # "front_left.mp4",
    # "front_left_center.mp4",
    # "front_right.mp4",
    # # "front_right_center.mp4", [Not part of KAIST dataset]
    "rear_left.mp4",
    "rear_right.mp4",
]:
    main("M-MULTI-SLOW-KAIST/" + input_path)

