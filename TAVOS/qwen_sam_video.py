from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image, ImageDraw, ImageFont, ImageTk
import torch
import os
import gzip, pickle

import re

import sys

import json

import numpy as np
import cv2
import gc

# Add SAM3 module to the Python path so its submodules can be imported
sys.path.append('./sam3')
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_video_model


import matplotlib.pyplot as plt


def extract_json_array(text):
    """
    Extract and parse the first JSON array found in a raw text string.

    This is used to recover structured JSON output from the language model's
    response, which may include surrounding prose or markdown formatting.

    Args:
        text (str): Raw text potentially containing a JSON array.

    Returns:
        list | None: Parsed JSON array if found and valid, otherwise None.
    """
    # Find the outermost '[' ... ']' delimiters
    start_index = text.find('[')
    end_index = text.rfind(']')
    
    if start_index != -1 and end_index != -1 and start_index < end_index:
        json_string = text[start_index : end_index + 1]
        
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"Invalid json: {e}")
            return None
    else:
        print("'[' y ']' not found.")
        return None

# Maximum number of seed detections (per object) passed to SAM3 for propagation
TOP_K = 3

# IoU threshold used to decide whether two tracks or masks overlap significantly
IOU_THRESHOLD = 0.7

def compute_iou(mask1, mask2):
    """
    Compute the Intersection over Union (IoU) between two binary masks.

    Args:
        mask1 (np.ndarray): First binary mask (bool or 0/1 values).
        mask2 (np.ndarray): Second binary mask (bool or 0/1 values).

    Returns:
        float: IoU score in [0, 1]. Returns 0 if the union is empty.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def mask_to_bbox(mask):
    """
    Convert a segmentation mask tensor to an axis-aligned bounding box.

    The mask is transferred from GPU to CPU and converted to a uint8 numpy
    array before calling OpenCV's boundingRect.

    Args:
        mask (torch.Tensor): Binary mask tensor (may be on GPU).

    Returns:
        tuple | None: Bounding box as (x, y, w, h) in pixel coordinates,
                      or None if the mask is empty.
    """
    mask_np = mask.cpu().numpy().astype(np.uint8).squeeze()
    if not np.any(mask_np): return None
    return cv2.boundingRect(mask_np) # x, y, w, h


def resolve_tracks(confirmed_objects, iou_thresh=0.7):
    """
    Remove duplicate tracks using Non-Maximum Suppression (NMS) based on IoU.

    When the same physical object is tracked multiple times (e.g. from different
    seed frames), this function keeps only the highest-scoring track among those
    that overlap significantly across shared frames.

    The algorithm:
      1. Flatten all tracks from all objects into a single list.
      2. Sort tracks by score in descending order (higher score = higher priority).
      3. For each track i, suppress any lower-scored track j whose mean IoU
         with i across shared frames exceeds iou_thresh.
      4. Reconstruct the output dictionary preserving only surviving tracks.

    Args:
        confirmed_objects (dict): Mapping from object name (str) to a list of
            track dicts. Each track dict must have:
                - 'track_id' (str)
                - 'content' (list of dicts with 'frame_id' and 'mask' keys)
                - 'score' (float, optional — defaults to 0.0)
        iou_thresh (float): IoU threshold above which a lower-scored track is
            suppressed. Default is 0.7.

    Returns:
        dict: Same structure as confirmed_objects, but with duplicate tracks
              removed. Each surviving track contains 'track_id', 'content',
              and 'score'.
    """
    all_tracks = []
    
    for obj_name, tracks in confirmed_objects.items():
        for t in tracks:
            # Work on a shallow copy to avoid mutating the original dict
            t_work = t.copy()
            t_work["obj_name"] = obj_name
            # Ensure score is a float for reliable sorting
            t_work["priority_score"] = float(t.get("score", 0.0))
            all_tracks.append(t_work)

    # Sort all tracks globally by score, highest first
    all_tracks = sorted(all_tracks, key=lambda x: x["priority_score"], reverse=True)
    
    suppressed_indices = set()
    n = len(all_tracks)

    for i in range(n):
        if i in suppressed_indices: continue
        ti = all_tracks[i]
        
        # Build a frame_id → mask lookup for the reference track
        masks_i = {c["frame_id"]: c["mask"] for c in ti["content"]}

        for j in range(i + 1, n):
            if j in suppressed_indices: continue
            tj = all_tracks[j]
            
            # Only compare tracks that share at least one frame
            common_frames = set(masks_i.keys()) & set(c["frame_id"] for c in tj["content"])
            if not common_frames: continue

            mj_dict = {c["frame_id"]: c["mask"] for c in tj["content"]}
            ious = [compute_iou(masks_i[f], mj_dict[f]) for f in common_frames]
            mean_iou = sum(ious) / len(ious)

            if mean_iou > iou_thresh:
                # Track j is suppressed because it has a lower score than track i
                suppressed_indices.add(j)

    # Rebuild the output dict with only non-suppressed tracks
    final_output = {obj: [] for obj in confirmed_objects.keys()}
    
    for idx, t in enumerate(all_tracks):
        if idx not in suppressed_indices:
            clean_track = {
                "track_id": t["track_id"],
                "content": t["content"],
                "score": t["priority_score"]
            }
            final_output[t["obj_name"]].append(clean_track)
            
    return final_output

# Default colour palette for mask overlays (RGB float values in [0, 1]).
# Each object class is assigned one colour cyclically.
DEFAULT_COLORS = [
    (0.0, 1.0, 0.4),   # green
    (1.0, 0.3, 0.3),   # red
    (0.0, 0.7, 1.0),   # cyan-blue
    (1.0, 0.8, 0.0),   # yellow
    (0.8, 0.0, 1.0),   # purple
    (0.0, 1.0, 1.0),   # cyan
]

def overlay_masks_on_frames(
    frames: list[Image.Image],
    confirmed_objects_resolved: dict,
    alpha: float = 0.4,
    colors: list[tuple] = DEFAULT_COLORS,
) -> list[Image.Image]:
    """
    Composite segmentation masks onto a list of PIL frames as semi-transparent overlays.

    Each object class is assigned a unique colour from the palette. For every
    frame, all masks belonging to confirmed tracks are blended over the original
    image using RGBA alpha compositing.

    Args:
        frames (list[Image.Image]): Original video frames in order.
        confirmed_objects_resolved (dict): Output of resolve_tracks(). Maps object
            name → list of track dicts, each containing 'content' with per-frame
            'mask' (np.ndarray, uint8) and 'frame_id' entries.
        alpha (float): Opacity of the mask overlay (0 = transparent, 1 = opaque).
            Default is 0.4.
        colors (list[tuple]): Colour palette as (R, G, B) float tuples in [0, 1].
            Object classes are assigned colours cyclically.

    Returns:
        list[Image.Image]: List of RGB frames with masks composited on top,
                           in the same order as the input frames.
    """
    n_frames = len(frames)

    # Assign one consistent colour per object class for temporal coherence
    object_color = {
        name: colors[i % len(colors)]
        for i, name in enumerate(confirmed_objects_resolved.keys())
    }

    # Build a per-frame index: frame_id → list of (mask, color) pairs
    frame_masks: dict[int, list[tuple[np.ndarray, tuple]]] = {i: [] for i in range(n_frames)}

    for obj_name, tracks in confirmed_objects_resolved.items():
        color = object_color[obj_name]
        for track in tracks:
            for entry in track["content"]:
                fid = entry["frame_id"]
                if fid < n_frames:
                    frame_masks[fid].append((entry["mask"], color))

    # Compose each frame by blending all its associated masks
    result = []
    for fid, frame in enumerate(frames):
        base = frame.convert("RGBA")
        h, w = np.array(frame).shape[:2]

        for mask, color in frame_masks[fid]:
            # Build an RGBA layer: coloured where the mask is active, transparent elsewhere
            mask_rgba = np.zeros((h, w, 4), dtype=np.float32)
            mask_rgba[mask > 0] = [*color, alpha]
            # Convert to a PIL RGBA image and alpha-composite over the base frame
            mask_layer = Image.fromarray((mask_rgba * 255).astype(np.uint8), mode="RGBA")
            base = Image.alpha_composite(base, mask_layer)

        result.append(base.convert("RGB"))

    return result


if __name__ == "__main__":

    # -----------------------------------------------------------------
    # 1. Configuration — paths and task description
    # -----------------------------------------------------------------

    # Local path to the Qwen vision-language model
    model_local_path = "D:/code/qwen35/Qwen3.5-4B"
    # model_local_path = "./Qwen3.5-4B"

    processor = AutoProcessor.from_pretrained(model_local_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_local_path, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )

    # Directory containing the input video frames (JPEG files, sorted by name)
    frames_path = './P01-20240202-161948--P01_R03_S04__occ1__chunk_1--20823_20867'

    # Paths for intermediate and final outputs
    objects_filename_text = './objects.txt'   # Raw LLM response
    objects_filename_json = './objects.json'  # Parsed object list

    masks_output = './masks.pkl.gz'           # Compressed pickle with all track data
    demo_output  = './test'                   # Directory for visualisation frames

    # Natural-language description of the cooking step to focus on
    task_name = 'Once your cacio e pepe has become a creamy, silky sauce, add your piping hot pasta to the bowl and toss vigorously to coat.'

    # Prompt sent to the VLM: ask for task-relevant objects with relevance grades
    prompt = "Based on these sequential frames, please provide a list of objects contained in the sequence but only the objects which are related with the following task \'" + task_name + "\'. Also provide a grade for each object, higher grade implies more related with the task, (e.g 10 means strongly related). Make sure that the result is in .json format (no bounding-box)"

    # -----------------------------------------------------------------
    # 2. Load frames and run the vision-language model (Qwen)
    # -----------------------------------------------------------------

    # Collect sorted frame names (without extension) and full file paths
    frame_names = sorted([f.split('.')[0] for f in os.listdir(frames_path) if f.endswith(".jpg")])
    frames_filename_path = sorted([os.path.join(frames_path,f) for f in os.listdir(frames_path) if f.endswith(".jpg")])

    # Load all frames and downsample to half resolution to reduce GPU memory usage
    frames = [Image.open(local_path) for local_path in frames_filename_path]
    frames_red = [frame.resize((frame.width // 2, frame.height // 2), Image.Resampling.LANCZOS) for frame in frames]   

    # Build the multimodal message: one image entry per frame, followed by the text prompt
    msg_content = []
    for frame in frames_red:
        msg_content.append({"type": "image", "image": frame})
    msg_content.append({"type": "text", "text": prompt})

    messages = [
        {
            "role": "user",
            "content": msg_content,
        }
    ]

    # Apply the model's chat template to format the input correctly
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    # Tokenise and move inputs to the model's device (GPU if available)
    inputs = processor(text=[text], images=frames_red, return_tensors="pt").to(model.device)

    # Run greedy decoding (do_sample=False) to get a deterministic response
    generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False)

    # Trim the prompt tokens from the output so we only decode the generated part
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)

    # Persist the raw LLM response for inspection
    with open(objects_filename_text, "w", encoding="utf-8") as f:
        f.write(output_text[0])
    print(output_text[0])

    # Extract the JSON array from the model's response and save it
    parsed_data = extract_json_array(output_text[0])
    qwen_data = {'task_name': task_name, 'objects': parsed_data}
    with open(objects_filename_json, "w") as f:
        json.dump(qwen_data, f, indent=4)

    # -----------------------------------------------------------------
    # 3. Free GPU memory before loading SAM3
    # -----------------------------------------------------------------

    del generated_ids_trimmed, generated_ids, inputs, text, messages, msg_content
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # 4. Load SAM3 models (video tracker + per-image grounded segmenter)
    # -----------------------------------------------------------------

    sam3_video = build_sam3_video_model()
    predictor = sam3_video.tracker
    # Share the detector backbone with the tracker to avoid loading it twice
    predictor.backbone = sam3_video.detector.backbone

    img_model = build_sam3_image_model()
    processor_image = Sam3Processor(img_model, confidence_threshold=0.5)

    # -----------------------------------------------------------------
    # 5. Filter Qwen objects by relevance grade
    # -----------------------------------------------------------------

    # Only process objects whose grade exceeds this threshold
    grade_threshold = 5
    if qwen_data['objects'] is not None:
        if len(qwen_data['objects']) > 0 and ('object' in qwen_data['objects'][0]):
            valid_object_list = [obj["object"] for obj in qwen_data.get("objects", []) if obj["grade"] > grade_threshold]
        else:
            valid_object_list = []
            print('error')
    else:
        valid_object_list = []
        print(' no object in : ' + frames_path)

    # -----------------------------------------------------------------
    # 6. Per-frame grounded detection → seed candidate masks
    # -----------------------------------------------------------------

    frame_filenames = sorted(os.listdir(frames_path))
    num_total_frames = len(frame_filenames)

    # Read the first frame to obtain the native video resolution
    first_frame = Image.open(os.path.join(frames_path, frame_filenames[0]))
    w, h = first_frame.size  # PIL returns (width, height)

    # Initialise the SAM3 video inference state for this clip
    inference_state = predictor.init_state(video_path=frames_path)

    # confirmed_objects_format: object_name → list of track dicts
    confirmed_objects_format = {}

    for obj in valid_object_list:
        mask_candidates = []
        confirmed_objects_format[obj] = []
        
        # Run grounded image segmentation on every frame to collect mask candidates
        for f_idx, f_name in enumerate(frame_filenames):
            image = Image.open(os.path.join(frames_path, f_name)).convert("RGB")
            inference_state_image = processor_image.set_image(image)
            output = processor_image.set_text_prompt(state=inference_state_image, prompt=obj)

            masks  = output.get("masks")
            scores = output.get("scores")
            
            if masks is not None:
                for m, s in zip(masks, scores):
                    bbox = mask_to_bbox(m)
                    if bbox:
                        mask_candidates.append({
                            "frame": f_idx,
                            "bbox": bbox,      # (x, y, w, h) in pixels
                            "score": float(s),
                            "mask_raw": m
                        })

        # Keep only the TOP_K highest-scoring detections as propagation seeds
        mask_to_propagate = sorted(mask_candidates, key=lambda x: x["score"], reverse=True)[:TOP_K]
        remaining_seeds = list(mask_to_propagate)

        # -----------------------------------------------------------------
        # 7. Propagate each seed through the video with SAM3
        # -----------------------------------------------------------------

        while remaining_seeds:
            seed = remaining_seeds.pop(0)
            frame_local = seed["frame"]
            bbox = seed["bbox"]  # (x, y, w, h)

            # Normalise the bounding box to relative coordinates [x1, y1, x2, y2]
            # as required by SAM3's add_new_points_or_box interface
            rel_box = np.array([[
                bbox[0] / w,
                bbox[1] / h,
                (bbox[0] + bbox[2]) / w,
                (bbox[1] + bbox[3]) / h
            ]], dtype=np.float32)

            # Reset any previous points/boxes and register the new seed box
            predictor.clear_all_points_in_video(inference_state)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_local,
                obj_id=1,
                box=rel_box
            )

            video_segments = {}  # frame_id → binary mask (np.ndarray)
            max_frames = inference_state["num_frames"]

            # Forward pass: propagate from the seed frame to the last frame
            max_forward = (num_total_frames - 1) - frame_local
            for f_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=frame_local,
                max_frame_num_to_track=max_forward,
                reverse=False,
                propagate_preflight=True
            ):
                video_segments[f_idx] = (video_res_masks[0] > 0).cpu().numpy()

            # Backward pass: propagate from the seed frame to the first frame
            max_backward = frame_local
            if max_backward > 0:
                for f_idx, obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=frame_local,
                    max_frame_num_to_track=max_backward,
                    reverse=True,
                    propagate_preflight=False
                ):
                    video_segments[f_idx] = (video_res_masks[0] > 0).cpu().numpy()

            # Sort segments by frame index for consistent downstream processing
            video_segments = dict(sorted(video_segments.items()))

            # Remove remaining seeds that are already covered by this propagated track
            # (i.e. their detection bbox overlaps sufficiently with the propagated mask)
            new_remaining = []
            for other_seed in remaining_seeds:
                f_other = other_seed["frame"]
                if f_other in video_segments:
                    prop_mask = video_segments[f_other]
                    if not np.any(prop_mask):
                        # No mask at this frame → seed is not covered, keep it
                        new_remaining.append(other_seed)
                        continue
                    
                    # Approximate the other seed's mask as a filled bounding-box region
                    seed_mask = np.zeros((h, w), dtype=bool)
                    ox, oy, ow, oh = other_seed["bbox"]
                    seed_mask[oy:oy+oh, ox:ox+ow] = True
                    
                    iou = compute_iou(prop_mask.astype(bool), seed_mask)
                    if iou < IOU_THRESHOLD:
                        # Low overlap → the other seed covers a different region, keep it
                        new_remaining.append(other_seed)
                else:
                    # The propagated track does not reach this frame → keep the seed
                    new_remaining.append(other_seed)
            remaining_seeds = new_remaining

            # Collect per-frame mask data for the new track
            track_masks_data = []
            for f_idx in sorted(video_segments.keys()):
                m_np = video_segments[f_idx]
                
                if np.any(m_np):
                    # Convert bool array to uint8 so OpenCV functions work correctly
                    if m_np.dtype == bool:
                        m_np = m_np.astype(np.uint8)
                    
                    # Ensure the mask is strictly 2-D (H, W) by squeezing singleton dims
                    m_np = m_np.squeeze()
                    
                    track_masks_data.append({
                        "frame_id": f_idx,
                        "mask": m_np,
                        "bbox": cv2.boundingRect(m_np)
                    })

            # Register the new track in the output dictionary
            new_track = {
                "track_id": f"pred_{len(confirmed_objects_format[obj])}",
                "content": track_masks_data,  # list of {frame_id, mask, bbox}
                "score": seed["score"]
            }
            confirmed_objects_format[obj].append(new_track)

    # -----------------------------------------------------------------
    # 8. Resolve duplicate tracks and save results
    # -----------------------------------------------------------------

    # Apply NMS across all tracks to remove duplicates caused by multiple seeds
    confirmed_objects_resolved = resolve_tracks(confirmed_objects_format)

    final_data = {
        "video_id": frames_path,
        "task": task_name,
        "confirmed_objects": confirmed_objects_resolved,
        "uncertain_objects": {}
    }

    # Serialise the full tracking output as a compressed pickle file
    with gzip.open(masks_output, 'wb') as f:
        pickle.dump(final_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # -----------------------------------------------------------------
    # 9. Render and save visualisation frames
    # -----------------------------------------------------------------

    # Composite segmentation masks onto the original (full-resolution) frames
    fused_frames = overlay_masks_on_frames(frames, confirmed_objects_resolved, alpha=0.4, colors=DEFAULT_COLORS)

    # Save each annotated frame to the demo output directory
    for frame, name in zip(fused_frames, frame_names):
        # Append .png extension if the name has none
        if not os.path.splitext(name)[1]:
            name = name + ".png"
        frame.save(os.path.join(demo_output, name))
