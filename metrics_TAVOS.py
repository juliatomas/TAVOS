import gzip
import pickle
import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
import sys
import types
import copy

# Import your established logic to keep things DRY (Don't Repeat Yourself)
from metrics_TAOS import (
    TAOSMetricTracker, 
    SemanticEvaluator, 
    prepare_dual_gt, 
    evaluate_agnostic_frame_max_iou, 
    show_comparison,
    show_comparison_individual,
    load_pkl_gz,
    clean_redundant_predictions
)



def get_asymmetric_frame_snapshot(gt_objects, pred_objects, target_frame_id):
    """
    GT: Colapsado por etiqueta (Smoothie).
    PRED: Lista de máscaras individuales de todos los tracks (Instancias puras).
    """
    # --- A. COLLAPSE GT (Many-to-One target) ---
    collapsed_gt = {}
    for label, tracks in gt_objects['confirmed_objects'].items():
        combined_mask = None
        state = "unknown"
        for track in tracks:
            state = track.get('state', 'explicit') 
            for content in track.get('content', []):
                if content['frame_id'] == target_frame_id:
                    m = content['mask'].astype(bool)
                    combined_mask = m if combined_mask is None else (combined_mask | m)
        
        if combined_mask is not None:
            collapsed_gt[label] = {
                "id": f"gt_{label}", "name": label, 
                "mask": combined_mask, "state": state
            }

    if pred_objects == None:
        return list(collapsed_gt.values()), None
    
    # --- B. KEEP PREDS SEPARATE (Individual evidence) ---
    individual_preds = []
    for label, tracks in pred_objects['confirmed_objects'].items():
        for track in tracks:
            t_id = track.get('track_id', 'unknown')
            for content in track.get('content', []):
                if content['frame_id'] == target_frame_id:
                    individual_preds.append({
                        "id": f"pred_{label}_{t_id}", 
                        "label": label,
                        "mask": content['mask'].astype(bool)
                    })

    return list(collapsed_gt.values()), individual_preds

class TAVOSMetricTracker(TAOSMetricTracker):
    def __init__(self, iou_threshold=0.4, semantic_threshold=0.85):
        super().__init__(iou_threshold, semantic_threshold)
        
        # --- Video-Specific Accumulators ---
        # Structure: { sequence_id: { object_id: [iou_frame1, iou_frame2, ...] } }
        self.sequence_data = {}
        

    def update_video_frame(self, sequence_id, metrics, assignments, hallucinations, 
                           gt_list, pred_list, shape, semantic_results=None, debug=False):
        """
        Updates metrics for a specific frame within a video sequence.
        """
        # 1. Standard frame update (Updates global counts, micro-miou, and image-miou)
        # We call the parent update to keep all existing image-level logic
        super().update(metrics, assignments, hallucinations, gt_list, pred_list, 
                       shape, semantic_results, debug)

        # 2. Sequence-specific Tracking
        if sequence_id not in self.sequence_data:
            self.sequence_data[sequence_id] = {}

        for gt_id, data in metrics.items():
            # Skip ambiguous objects as per our previous logic
            if data.get('state') == 'ambiguous':
                continue
            
            if gt_id not in self.sequence_data[sequence_id]:
                self.sequence_data[sequence_id][gt_id] = {
                    "ious": [],
                    "state": data.get('state', 'explicit')
                }
            
            # Extract IoU (handles both dict and float formats)
            iou_val = data["iou"] if isinstance(data, dict) else data
            self.sequence_data[sequence_id][gt_id]["ious"].append(iou_val)

    def report_video_results(self):
        """
        Calculates and prints the TAVOS-specific results (Track-based J-mean).
        """
        all_track_ious = []
        tier_track_ious = {t: [] for t in self.tiers}

        # Calculate average IoU per object track
        for seq_id, objects in self.sequence_data.items():
            for obj_id, info in objects.items():
                track_avg = np.mean(info["ious"])
                state = info["state"]
                
                all_track_ious.append(track_avg)
                if state in self.tiers:
                    tier_track_ious[state].append(track_avg)

        # Calculate Final J-mean (The standard VOS metric)
        j_mean = np.mean(all_track_ious) if all_track_ious else 0

        print("\n" + "="*65)
        print(f"{'TAVOS VIDEO EVALUATION REPORT (SEQUENCE-BASED)':^65}")
        print("="*65)
        
        # 1. Overall Track Performance
        print(f"{'OVERALL VIDEO PERFORMANCE (J-mean)':<40}")
        print(f"  Sequence mIoU (Track J-mean):      {j_mean:.4f}")
        print(f"  Total Video Tracks Evaluated:      {len(all_track_ious)}")
        
        # 2. Tier Breakdown (at the Track Level)
        print("-" * 65)
        print(f"{'TIER-WISE TRACK BREAKDOWN':<40} {'J-mean':<10}")
        print("-" * 65)
        for t in self.tiers:
            t_j_mean = np.mean(tier_track_ious[t]) if tier_track_ious[t] else 0
            print(f"  {t.capitalize():<38} {t_j_mean:.4f}")

        # 3. Call standard TAOS report for frame-level and semantic context
        print("-" * 65)
        print("  [FRAME-LEVEL SUMMARY FOR REFERENCE]")
        super().report()

    def save_report_video(self, filepath):
        """Generates and saves the full benchmark report to a text file and prints LaTeX rows."""
        # --- 1. Image-Level Spatial Metrics ---
        precision = self.total_assignments / self.total_pred_count if self.total_pred_count > 0 else 0
        recall = self.total_assignments / self.total_gt_count if self.total_gt_count > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        macro_miou = np.mean(self.image_mious) if self.image_mious else 0
        global_pixel_iou = self.global_inter_sum / self.global_union_sum if self.global_union_sum > 0 else 0
        
        # Individual Image Tiers (mIoU)
        m_exp = np.mean(self.tier_image_mious['explicit']) if self.tier_image_mious['explicit'] else 0
        m_imp = np.mean(self.tier_image_mious['implicit']) if self.tier_image_mious['implicit'] else 0
        m_can = np.mean(self.tier_image_mious['candidate']) if self.tier_image_mious['candidate'] else 0

        # Task-Aware mIoU (TA-mIoU)
        # Weights: Explicit (1.0), Implicit (1.0), Candidate (0.1)
        ta_miou = (m_exp * 1.0 + m_imp * 1.0 + m_can * 0.1) / 2.1

        # --- 2. Video-Level Metrics (Track-based J-mean) ---
        all_track_ious = []
        tier_track_ious = {t: [] for t in self.tiers}
        for seq_id, objects in self.sequence_data.items():
            for obj_id, info in objects.items():
                track_avg = np.mean(info["ious"])
                all_track_ious.append(track_avg)
                if info["state"] in self.tiers:
                    tier_track_ious[info["state"]].append(track_avg)
        
        j_tot = np.mean(all_track_ious) if all_track_ious else 0
        j_exp = np.mean(tier_track_ious['explicit']) if tier_track_ious['explicit'] else 0
        j_imp = np.mean(tier_track_ious['implicit']) if tier_track_ious['implicit'] else 0
        j_can = np.mean(tier_track_ious['candidate']) if tier_track_ious['candidate'] else 0

        # --- 3. Reliability & Reasoning Metrics ---
        # Task Hallucination Rate (Percentage of FPs over total predictions)
        thr = (self.total_hallucinations / self.total_pred_count * 100 if self.total_pred_count > 0 else 0)
        # Average False Positives per frame
        fp_img = self.total_hallucinations / len(self.image_mious) if self.image_mious else 0
        # Semantic Consistency (Macro average)
        sc = np.mean(self.image_sc_ratios) if self.image_sc_ratios else 0

        # --- 4. WRITE TO .TXT FILE ---
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*65 + "\n")
            f.write(f"{'TAVOS FINAL BENCHMARK REPORT':^65}\n")
            f.write("="*65 + "\n\n")
            f.write(f"Global Pixel IoU: {global_pixel_iou:.3f} | Macro mIoU: {macro_miou:.3f} | TA-mIoU: {ta_miou:.3f}\n")
            f.write(f"J-Total (Video):  {j_tot:.3f}\n\n")
            
            f.write("-" * 65 + "\n")
            f.write(f"{'TIER BREAKDOWN':<30} {'mIoU (Img)':<12} | {'J-mean (Vid)':<12}\n")
            f.write("-" * 65 + "\n")
            f.write(f"  Explicit                     {m_exp:.3f}       | {j_exp:.3f}\n")
            f.write(f"  Implicit                     {m_imp:.3f}       | {j_imp:.3f}\n")
            f.write(f"  Candidate                    {m_can:.3f}       | {j_can:.3f}\n\n")

            f.write(f"{'REASONING & RELIABILITY':<40}\n")
            f.write(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}\n")
            f.write(f"  THR: {thr:.2f}% | FP/img: {fp_img:.3f} | SC: {sc:.3f}\n")

        # --- 5. GENERATE LATEX ROWS (Print to Console) ---
        print("\n" + "#"*20 + " COPY FOR LATEX " + "#"*20)
        # Row 1: Main Results Table
        # Format: G-IoU & mIoU & mExp & mImp & mCand & TA-m & JExp & JImp & JCand & JTot
        row1 = (f"{global_pixel_iou:.3f} & {macro_miou:.3f} & {m_exp:.3f} & {m_imp:.3f} & "
                f"{m_can:.3f} & {ta_miou:.3f} & {j_exp:.3f} & {j_imp:.3f} & {j_can:.3f} & {j_tot:.3f} \\\\")
        print("\nTABLE 1 (Main Results):")
        print(row1)

        # Row 2: Diagnostics Table
        # Format: Precision & Recall & F1 & THR & FP/img & SC
        row2 = f"{precision:.3f} & {recall:.3f} & {f1:.3f} & {thr:.2f} & {fp_img:.3f} & {sc:.3f} \\\\"
        print("\nTABLE 2 (Diagnostics):")
        print(row2)
        print("#"*56 + "\n")

        return True
def parse_sa2va_to_tavos_format(sa2va_masks, chunk_id, task_name):
    """
    Parses a monolithic Sa2VA mask array into the TAVOS dictionary format.
    
    Args:
        sa2va_masks (np.array): Boolean array of shape (N, H, W)
        chunk_id (str): The ID/name of the video chunk
        task_name (str): The text prompt/task associated with the chunk
        
    Returns:
        dict: Data structured for the TAVOS evaluation pipeline
    }
    """
    num_frames = sa2va_masks.shape[0]
    
    # Initialize the frame content list
    track_content = []
    
    for i in range(num_frames):
        mask_frame = sa2va_masks[i].astype(np.uint8)
        
        # Calculate Bounding Box from the mask pixels
        y_indices, x_indices = np.where(mask_frame > 0)
        if len(x_indices) > 0:
            bbox = [
                int(np.min(x_indices)), 
                int(np.min(y_indices)), 
                int(np.max(x_indices)), 
                int(np.max(y_indices))
            ]
        else:
            # Fallback if no mask pixels exist in this frame
            bbox = [0, 0, 0, 0]
            
        # Create frame dictionary
        frame_data = {
            'frame_id': i,
            'mask': mask_frame,
            'bbox': bbox
        }
        track_content.append(frame_data)
        
    # Build the hierarchical structure
    parsed_data = {
        'video_id': chunk_id,
        'task': task_name,
        'confirmed_objects': {
            'object': [
                {
                    'track_id': 'track_0',
                    'score': 0.0,
                    'content': track_content
                }
            ]
        }
    }
    
    return parsed_data

def run_video_evaluation(gt_dir, pred_dir, img_dir, limit_chunks=None, baseline='ours'):
    """
    Video evaluation pipeline. 
    Iterates through video chunks, then through frames within each chunk.
    """

    save_video = False


    # 1. Chunk Discovery
    # Each pkl.gz represents a full video sequence/chunk
    all_chunks = sorted(glob.glob(os.path.join(gt_dir, "*.pkl.gz")))
    
    if limit_chunks:
        all_chunks = all_chunks[:limit_chunks]

    # Initialize Video Tracker and Semantic Evaluator
    tracker = TAVOSMetricTracker(iou_threshold=0.4)
    eval_semantic = SemanticEvaluator()

    continue_eval = True

    # 2. Outer Loop: Iterate over Video Chunks (Sequences)
    for chunk_path in tqdm(all_chunks, desc="Evaluating Video Chunks"):
        chunk_id = os.path.basename(chunk_path).replace(".pkl.gz", "")
        chunk_id = os.path.basename(chunk_id).replace("GT_", "")

        
        if save_video:
            video_writer = None
            output_video_dir = pred_dir + '/videos'
            if not os.path.exists(output_video_dir):
                os.makedirs(output_video_dir)
        
        # Load Chunk Data (contains multiple frames)
        gt_chunk_data = load_pkl_gz(chunk_path)

        # print(gt_chunk_data['task'])

        ##### each gt_chunk_data has;
        # - gt_chunk_data['video_id']: 'name_of_the_video'
        # - gt_chunk_data['task']: 'task text prompt used in TAOS'
        # - gt_chunk_data['confirmed_objects']: dictionary, containing object labels as keys():
        # -- gt_chunk_data['confirmed_objects']['object1']: list of tracks of different masks assigned to the same object label
        # --- gt_chunk_data['confirmed_objects']['object1'][0] (for first track): dictionary, containing:
        # ---- gt_chunk_data['confirmed_objects']['object1'][0]['track_id']: id of the track
        # ---- gt_chunk_data['confirmed_objects']['object1'][0]['state']: whether it is implicit, explicit, candidate, ambiguous
        # ---- gt_chunk_data['confirmed_objects']['object1'][0]['score']: not used in practice
        # ---- gt_chunk_data['confirmed_objects']['object1'][0]['content']: list of mask contents per each mask:
        # ----- gt_chunk_data['confirmed_objects']['object1'][0]['content'][0] (for frame 0): 
        # ------ gt_chunk_data['confirmed_objects']['object1'][0]['content'][0]['frame_id']: numeric id of the frame
        # ------ gt_chunk_data['confirmed_objects']['object1'][0]['content'][0]['mask']: numpy array with the mask
        # ------ gt_chunk_data['confirmed_objects']['object1'][0]['content'][0]['bbox']: bounding box of the mask

        # --- Identify FRAMES ---
        all_frame_ids = set()
        for label, tracks in gt_chunk_data['confirmed_objects'].items():
            for track in tracks:
                for content in track['content']:
                    all_frame_ids.add(content['frame_id'])
        sorted_frames = sorted(list(all_frame_ids))
        
        if baseline == 'ActionVOS':
            pred_path = os.path.join(pred_dir, f"{chunk_id}.pkl").replace("GT_","")
        elif baseline == 'Sa2VA':
            pred_path = os.path.join(pred_dir, f"{chunk_id}.npy").replace("GT_","")
        else:
            pred_path = os.path.join(pred_dir, f"{chunk_id}.pkl.gz").replace("GT_","")



        video_id = gt_chunk_data['video_id']
        task_name = gt_chunk_data.get('task', 'Unknown Task')

        if not os.path.exists(pred_path):
            print(f"Skipping chunk {chunk_id}: Prediction file not found.")
            # continue
            pred_chunk_data = None
        else:
            if baseline == 'ActionVOS':
                with open(pred_path, 'rb')  as f:
                    pred_chunk_data = pickle.load(f)
                pred_chunk_data['confirmed_objects'].update(pred_chunk_data['uncertain_objects'])
            elif baseline == 'Sa2VA':
                sa_masks = np.load(pred_path, allow_pickle=True)
                pred_chunk_data = parse_sa2va_to_tavos_format(sa_masks, gt_chunk_data['video_id'], gt_chunk_data['task'])
            else:
                pred_chunk_data = load_pkl_gz(pred_path)




        # 3. Inner Loop: Iterar frame a frame (Temporal dimension)
        for f_id in sorted_frames:
            gt_list_frame, pred_list_frame = get_asymmetric_frame_snapshot(gt_chunk_data, pred_chunk_data, f_id)

            if not gt_list_frame and not pred_list_frame:
                continue

            pred_list_frame, _ = clean_redundant_predictions(pred_list_frame, debug=False)
            dual_gt_list = prepare_dual_gt(gt_list_frame)

            img_filename = f"{f_id:06d}.jpg"
            img_path = os.path.join(img_dir, video_id, img_filename)
            image = cv2.imread(img_path)
            h, w = image.shape[:2] if image is not None else dual_gt_list[0]['mask_hollow'].shape

            # 4. Matching Many-to-One
            metrics, gt_entities, assignments, hallucinations = evaluate_agnostic_frame_max_iou(
                dual_gt_list, pred_list_frame, (h, w)
            )

            # 5. Semantic Consistency
            semantic_res = eval_semantic.compute_sc_metrics(assignments, debug=False)


            tracker.update_video_frame(
                sequence_id=video_id,
                metrics=metrics,
                assignments=assignments,
                hallucinations=hallucinations,
                gt_list=gt_list_frame,
                pred_list=pred_list_frame,
                shape=(h, w),
                semantic_results=semantic_res,
                debug=False
            )

            # --- VISUALIZATION ---
            # Show the triple-panel comparison (GT | PRED | ERROR MAP)
            # continue_eval, canvas = show_comparison(image, gt_list_frame, gt_entities, assignments, hallucinations, metrics, task_name, save = False, folder=pred_dir, filename=video_id + '_' + img_filename, stop_per_frame = False)
            # continue_eval, _ = show_comparison_individual(image, gt_list_frame, gt_entities, assignments, hallucinations, metrics, task_name, save = False, folder=pred_dir, filename=video_id + '_' + img_filename, stop_per_frame = True)
                # If the user presses 'q' or closes the window, stop the evaluation
                
            # # Initialize VideoWriter on the first frame of the chunk
            # if save_video:
            #     if video_writer is None:
            #         h_c, w_c = canvas.shape[:2]
            #         video_path = os.path.join(output_video_dir, f"{chunk_id}_eval.mp4")
            #         fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'XVID'
            #         video_writer = cv2.VideoWriter(video_path, fourcc, 5.0, (w_c, h_c))
            #     # Write the frame to the video file
            #     video_writer.write(canvas)

            if not continue_eval:    
                break

        # Clean up VideoWriter for this chunk
        if save_video:
            if video_writer:
                video_writer.release()
                print(f"Saved evaluation video: {video_id}_eval.mp4")
        
        if not continue_eval:    
            break


    # FINAL VIDEO REPORT
    # This will print the Track-based J-mean (VOS standard)
    tracker.report_video_results()
    tracker.save_report_image(os.path.join(pred_dir, "final_metrics_report_image.txt"))
    tracker.save_report_video(os.path.join(pred_dir, "final_metrics_report_video.txt"))
    cv2.destroyAllWindows()

if __name__ == "__main__":
    GT_CHUNK_DIR = '/meta_pkl'
    IMG_ROOT_DIR = 'P02_chunks_frames'
    PRED_CHUNK_DIR = './20_qwen4B_grade_sam3_prop'

    run_video_evaluation(GT_CHUNK_DIR, PRED_CHUNK_DIR, IMG_ROOT_DIR, limit_chunks=100, baseline='asdas')
