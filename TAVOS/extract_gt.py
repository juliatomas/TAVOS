import cv2          
import os           
import gzip         
import pickle       
import numpy as np  
import sys          
import gc           
import re           

# Compatibility patch: some older pickle files reference 'numpy._core', which doesn't
# exist in recent NumPy versions. This maps it to the actual 'numpy.core' module
# so that deserialization doesn't fail.
sys.modules['numpy._core'] = np.core


def process_a_video(ann_path, video_root, images_dir, meta_dir):
    """
    Processes a single annotated video segment:
      1. Loads annotation data from a compressed pickle file.
      2. Extracts and resizes frames from the corresponding video file.
      3. Retrieves object masks for each extracted frame (using a subsampled stride).
      4. Builds a ground-truth (GT) dictionary with bounding boxes and masks.
      5. Saves extracted frames as JPEG images and the GT metadata as a compressed pickle.

    Args:
        ann_path    (str): Path to the annotation file (.pkl.gz).
        video_root  (str): Root directory where raw video files (.mp4) are stored.
        images_dir  (str): Output directory where extracted frame images will be saved.
        meta_dir    (str): Output directory where GT metadata (.pkl.gz) will be saved.
    """

    # --- Load annotation file ---
    try:
        with gzip.open(ann_path, "rb") as f:
            ui_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {ann_path}: {e}")
        return  # Skip this file if loading fails

    # --- Extract metadata from the annotation ---
    video_id       = ui_data['video']['video_id']            # Unique video identifier (e.g., "P01-001")
    participant_id = video_id.split('-')[0]                  # Participant ID derived from the video ID (e.g., "P01")
    start_frame    = ui_data['video']['start_frame']         # First frame of the annotated segment
    end_frame      = ui_data['video']['end_frame']           # Last frame of the annotated segment
    step_id        = ui_data['video']['step_id']             # Identifier of the procedural step
    step_name      = ui_data['video']['step_name']           # Human-readable step name (may include parenthetical notes)

    # Preprocessing parameters used during annotation creation
    stride_original = ui_data['preprocess']['stride']        # Original frame sampling stride used during annotation
    target_shape    = ui_data['preprocess']['shape']         # Target (height, width) to resize frames to

    # --- Compute effective stride ---
    # We subsample further by a factor of new_stride on top of the original stride.
    # effective_stride is the total number of raw video frames to skip between saved frames.
    new_stride       = 10
    effective_stride = stride_original * new_stride

    # --- Clean up the step name ---
    # Remove trailing parenthetical content, e.g., "Cut onion (prep)" -> "Cut onion"
    task_clean = re.sub(r'\s*\([^()]+\)$', '', step_name).strip()

    # --- Build a unique folder ID for this video segment ---
    # Format: "<video_id>--<step_id>--<start_frame>_<end_frame>"
    folder_id = f"{video_id}--{step_id}--{start_frame}_{end_frame}"

    # --- Create the output image directory for this segment ---
    video_images_dest = os.path.join(images_dir, folder_id)
    os.makedirs(video_images_dest, exist_ok=True)

    # --- Initialize the ground-truth (GT) output structure ---
    # confirmed_objects: objects with high-confidence masks
    # uncertain_objects: objects with lower confidence (populated elsewhere if needed)
    final_gt = {
        'video_id':          folder_id,
        'task':              task_clean,
        'confirmed_objects': {},   # Dict: object_name -> list of track dicts
        'uncertain_objects': {}    # Dict: currently unused in this function
    }

    # --- Extract masks and object state annotations from the loaded data ---
    masks_data = ui_data.get('masks', {})                            # {obj_name: {track_id: {annot_idx: mask}}}
    obj_states = ui_data.get('prompts', {}).get('object_states', {}) # {obj_name: state_label}

    # --- Open the corresponding video file ---
    video_path = os.path.join(video_root, participant_id, f"{video_id}.mp4")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return  # Skip if the video file is unavailable

    # Seek to the starting frame of the annotated segment
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print(f"Processing: {folder_id}...")

    # --- Compute how many frames we will extract ---
    # We extract one frame every effective_stride raw frames, from start_frame to end_frame.
    num_steps = (end_frame - start_frame) // effective_stride + 1

    for i in range(num_steps):
        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break  # Stop if we've reached the end of the video or a read error occurs

        # Resize the frame to the target resolution (width, height) and save as JPEG
        frame = cv2.resize(frame, (target_shape[1], target_shape[0]))  # cv2 uses (width, height)
        cv2.imwrite(os.path.join(video_images_dest, f"{i:06d}.jpg"), frame)

        # Map this output frame index (i) to the annotation index in the original pkl data.
        # The annotation was created with stride_original; we are now subsampling by new_stride,
        # so the annotation index for frame i is i * new_stride.
        annot_idx = i * new_stride

        # --- Process masks for each annotated object ---
        for obj_name, tracks in masks_data.items():

            # Ensure a list exists for this object in the GT output
            if obj_name not in final_gt['confirmed_objects']:
                final_gt['confirmed_objects'][obj_name] = []

            for track_id, track_frames in tracks.items():
                # Look up the mask for this object track at the current annotation index
                mask = track_frames.get(annot_idx)

                # Only process masks that exist AND contain at least one non-zero pixel
                if mask is not None and np.any(mask):

                    # Force a full copy as uint8 so we don't hold a reference to the
                    # large original data structure (helps with memory management).
                    mask_uint8 = np.array(mask, dtype=np.uint8, copy=True)

                    # Compute the axis-aligned bounding box of the mask region.
                    # Returns (x, y, width, height) in pixel coordinates.
                    bbox = cv2.boundingRect(mask_uint8)

                    # --- Append to an existing track or create a new one ---
                    track_found = False
                    for existing_track in final_gt['confirmed_objects'][obj_name]:
                        if existing_track['track_id'] == track_id:
                            # Track already registered: just append this frame's data
                            existing_track['content'].append({
                                'frame_id': i,
                                'mask':     mask_uint8,
                                'bbox':     bbox
                            })
                            track_found = True
                            break

                    if not track_found:
                        # First time seeing this track: create a new entry
                        final_gt['confirmed_objects'][obj_name].append({
                            'track_id': track_id,
                            'content': [{
                                'frame_id': i,
                                'mask':     mask_uint8,
                                'bbox':     bbox
                            }],
                            'score': 1.0,                                    # Confidence score (max for confirmed objects)
                            'state': obj_states.get(obj_name, "unknown")     # Semantic state label (e.g., "cooked", "sliced")
                        })

        # --- Efficiently skip frames between saved samples ---
        # Instead of decoding, we use cap.grab() which only reads the compressed frame
        # without full decoding, making the skip much faster.
        if effective_stride > 1:
            for _ in range(effective_stride - 1):
                cap.grab()

    cap.release()  # Release the video file handle

    # --- Save the ground-truth metadata as a compressed pickle ---
    meta_output_path = os.path.join(meta_dir, f"GT_{folder_id}.pkl.gz")
    with gzip.open(meta_output_path, 'wb') as f:
        pickle.dump(final_gt, f, protocol=4)  # Protocol 4 supports large objects and Python 3.4+

    # --- Explicit memory cleanup ---
    # The annotation data and masks can be very large; explicitly delete them and
    # trigger garbage collection to avoid memory buildup across many videos.
    del ui_data
    del masks_data
    del final_gt
    gc.collect()


def main():
    """
    Entry point: iterates over all annotation files in the annotations directory
    and calls process_a_video() on each one.
    """

    # --- Directory configuration ---
    chunks_split_path = "./TAVOS_annotations"    # Directory containing per-segment annotation files (.pkl.gz)
    video_root        = "./HD-EPIC/Videos"       # Root directory of the raw MP4 videos
    output_base       = "./TAVOS_dataset"        # Root output directory

    images_dir = os.path.join(output_base, "images")    # Subdirectory for extracted frame images
    meta_dir   = os.path.join(output_base, "meta_pkl")  # Subdirectory for GT metadata files

    # Create output directories if they don't already exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(meta_dir,   exist_ok=True)

    # Collect and sort all annotation files (ensures deterministic processing order)
    all_files = sorted([f for f in os.listdir(chunks_split_path) if f.endswith(".pkl.gz")])

    # Process each annotation file one at a time
    for filename in all_files:
        ann_path = os.path.join(chunks_split_path, filename)
        process_a_video(ann_path, video_root, images_dir, meta_dir)
        gc.collect()  # Free memory after each video before processing the next


if __name__ == "__main__":
    main()