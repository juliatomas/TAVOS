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
from collections import defaultdict

import torch
from transformers import AutoProcessor, SiglipModel
from scipy.ndimage import binary_fill_holes

sys.modules['numpy._core'] = np.core

# # --- MOCKS FOR COMPATIBILITY WITH NUMPY 1.x / 2.x ---
# mock_core = types.ModuleType("numpy._core")
# mock_core.__path__ = []
# sys.modules["numpy._core"] = mock_core
# import numpy.core.multiarray as multiarray
# mock_core.multiarray = multiarray
# sys.modules["numpy._core.multiarray"] = multiarray
# import numpy.core.numeric as numeric
# mock_core.numeric = numeric
# sys.modules["numpy._core.numeric"] = sys.modules.get("numpy.core.numeric", numeric)

# ==========================================
# METRIC FUNCTIONS
# ==========================================

def calculate_iou(mask_a, mask_b):
    """Calcula el IoU entre dos matrices booleanas."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0
    return intersection / union

def evaluate_agnostic_frame_max_iou(dual_gt_list, pred_list, shape):
    """
    Evaluates predictions against a dual Ground Truth (GT) representation (hollow and solid).
    It performs a many-to-one assignment and selects the best IoU between the two GT versions.
    """
    
    # 1. Initialize assignment storage
    # Each GT instance is prepared to store aggregated prediction masks and metadata.
    # We keep both 'hollow' (donut) and 'solid' (full) versions to decide the best match later.
    assignments = {gt["id"]: {
        "mask": np.zeros(shape, dtype=bool), 
        "metadata": [], 
        "gt_name": gt["name"],
        "state": gt["state"],
        "mask_hollow": gt["mask_hollow"],
        "mask_solid": gt["mask_solid"],
        "best_gt_mask": None # This will be decided after all predictions are aggregated
    } for gt in dual_gt_list}
    
    threshold_iop = 0.4 # Minimum Intersection over Prediction to avoid weak matches
    hallucinations = []

    # --- MATCHING LOOP ---
    # We iterate through each prediction and find the most suitable GT "owner"
    if pred_list is not None:
        for p_item in pred_list:
            p_mask = p_item["mask"].astype(bool)
            p_area = p_mask.sum()
            if p_area == 0: continue
            
            best_match = None
            max_match_score = -1.0 # Score used to decide the best GT candidate
            
            for gt_inst in dual_gt_list:
                # Step A: Calculate intersection with the 'solid' version
                # Solid is used for matching because it encompasses the entire area of the object.
                inter = np.logical_and(p_mask, gt_inst["mask_solid"]).sum()
                if inter == 0: continue
                
                # Step B: Calculate a temporary IoU for matching purposes.
                # This prevents large containers (like a jar) from absorbing specific objects 
                # (like water) just because they physically overlap.
                union = p_area + gt_inst["mask_solid"].sum() - inter
                match_iou = inter / union
                
                # [Optional debug blocks]
                # if match_iou > 0:
                #     print(gt_inst['name'],p_item['label'])
                #     cv2.imshow("gt",gt_inst['mask_solid'].astype(np.uint8)*255)
                #     cv2.imshow("pred", p_mask.astype(np.uint8)*255)
                #     cv2.waitKey(0)
                #     hola=1

                # Select the GT candidate with the highest overlap score
                if match_iou > max_match_score:
                    max_match_score = match_iou
                    best_match = gt_inst["id"]
                    current_inter = inter # Store intersection for IoP validation
            
            # Once the best GT candidate is found, verify if the assignment is valid
            if best_match:
                # Calculate Intersection over Prediction (IoP)
                # This allows small fragments to be correctly assigned to a larger GT entity.
                iop = current_inter / p_area
                if iop > threshold_iop:
                    # Aggregate the prediction mask into the assigned GT "bucket" (Many-to-One)
                    assignments[best_match]["mask"] |= p_mask
                    assignments[best_match]["metadata"].append({
                        "label": p_item.get("label", "unknown"),
                        "bbox": p_item.get("bbox")
                    })
                else:
                    # If overlap is too weak, it's considered a hallucination (False Positive)
                    hallucinations.append(p_item)
            else:
                # No matching GT found at all
                hallucinations.append(p_item)

    # --- FINAL STEP: MAX-IOU DECISION ---
    # After aggregating all predictions for each GT, we decide which version 
    # (hollow or solid) results in a better score.
    metrics = {}
    gt_entities_for_viz = {}
    
    for gt_id, data in assignments.items():
        agg_pred_mask = data["mask"]
        
        # Calculate IoU against both topological interpretations
        iou_h = calculate_iou(data["mask_hollow"], agg_pred_mask)
        iou_s = calculate_iou(data["mask_solid"], agg_pred_mask)
        
        # Pick the version that yields the highest IoU
        if iou_h >= iou_s:
            best_iou = iou_h
            data["best_gt_mask"] = data["mask_hollow"]
            data["type_won"] = "hollow"
        else:
            best_iou = iou_s
            data["best_gt_mask"] = data["mask_solid"]
            data["type_won"] = "solid"
            
        metrics[gt_id] = {
            "iou": best_iou,
            "state": data["state"] 
        }
        # Store the winning mask for visualization (Error Map)
        gt_entities_for_viz[gt_id] = data["best_gt_mask"]
    
    return metrics, gt_entities_for_viz, assignments, hallucinations

def collapse_gt_by_label(gt_list):
    """
    Groups multiple ground truth instances of the same class into a single entity.
    This is useful for class-level evaluation instead of individual instance evaluation.
    """
    collapsed = {}
    
    # 1. Merge masks by label
    for gt in gt_list:
        # Get the class name using 'name' or 'label' keys as fallbacks
        name = gt.get('name', gt.get('label', 'obj'))
        state = gt.get('state', 'unknown')  # Get the state
        mask = gt["mask"].astype(bool)
        group_key = (name, state)           # Use a composite key
        
        if group_key not in collapsed:
            collapsed[group_key] = {
                "id": f"{name}_{state}",    # Unique ID including state
                "name": name, 
                "state": state,             # Store the state here
                "mask": mask.copy()
            }
        else:
            # Merge the current mask with the existing one using bitwise OR
            collapsed[group_key]["mask"] |= mask
            
    # --- 2. RECALCULATE BOUNDING BOXES FOR THE MERGED MASKS ---
    # Since the new mask combines multiple objects, we need a new box that encompasses all of them.
    for name, data in collapsed.items():
        # Find coordinates (y, x) of all True pixels in the merged mask
        y, x = np.where(data["mask"])
        
        if len(x) > 0 and len(y) > 0:
            # Define the bounding box as [x_min, y_min, x_max, y_max]
            data["bbox"] = [
                int(np.min(x)), 
                int(np.min(y)), 
                int(np.max(x)), 
                int(np.max(y))
            ]
        else:
            # Default empty box if no pixels are found
            data["bbox"] = [0, 0, 0, 0]
            
    # Return the collapsed entities as a list
    return list(collapsed.values())

def prepare_dual_gt(gt_list, iop_threshold=0.95):
    """
    Generates two versions of each Ground Truth mask: 'hollow' and 'solid'.
    This accounts for cases where one object contains another (e.g., food inside a pan),
    allowing the evaluation to be agnostic to the model's segmentation style.
    """
    dual_gt = []
    
    for i, gt in enumerate(gt_list):
        mask_raw = gt['mask'].astype(bool)
        
        # Initialize a mask to store all objects found 'inside' the current one
        contents_mask = np.zeros_like(mask_raw)

        # --- IMPROVEMENT: Fill holes to detect content in 'hollow' containers ---
        # This turns a 'ring' pan into a 'solid disc' pan just for the check
        detection_mask = binary_fill_holes(mask_raw)
        
        # Cross-reference the current object (i) with every other object (j) in the frame
        for j, other in enumerate(gt_list):
            if i == j: continue  # Do not compare an object with itself
            
            m_other = other['mask'].astype(bool)
            
            # Calculate intersection to determine containment
            inter = np.logical_and(detection_mask, m_other).sum()
            
            # If the 'other' object is significantly contained within 'mask_raw'
            # (based on the Intersection over Prediction threshold)
            if inter > (m_other.sum() * iop_threshold):
                # # [Optional debug blocks]
                # print(gt['name'],other['name'])
                # print(mask_raw.sum(), m_other.sum(), inter)
                # cv2.imshow("gt",mask_raw.astype(np.uint8)*255)
                # cv2.imshow("pred", m_other.astype(np.uint8)*255)
                # cv2.waitKey(0)
                # hola=1
                
                # Add this object's pixels to the cumulative contents of the current container
                contents_mask |= m_other

        # Append the dual representation
        dual_gt.append({
            "id": gt["id"],
            "name": gt.get("name", gt.get("label", "obj")), # Maintain original semantic label
            "state": gt.get("state"),
            # 'hollow' (Donut): The container pixels MINUS the contents pixels
            "mask_hollow":  mask_raw & ~contents_mask, 
            # 'solid' (Disc): The container pixels PLUS the contents pixels
            "mask_solid": mask_raw | contents_mask,   
            
        })
        
    return dual_gt

def clean_redundant_predictions(pred_list, iou_threshold=0.85, expansion_threshold=0.95, debug=False):
    """
    Refines the prediction list by removing redundant or overlapping masks.
    It handles two main cases:
    1. Deduplication: Removing nearly identical masks of the same class.
    2. Group Cleaning (Explanation Filter): Removing a large "container" mask if it 
       is already effectively explained by a set of smaller, more specific parts.
    """
    
    if pred_list == None or len(pred_list) < 2:
        return pred_list, False
    
    # Sort predictions by mask area from LARGEST to SMALLEST.
    # This allows us to evaluate if large masks are redundant "containers" for smaller ones.
    pred_list.sort(key=lambda x: x['mask'].sum(), reverse=True)
    
    to_remove = set()
    any_change = False
    
    for i in range(len(pred_list)):
        large = pred_list[i]
        l_mask = large['mask']
        l_area = l_mask.sum()
        l_label = large['label']
        
        # --- 1. DEDUPLICATION (Same class, nearly same shape) ---
        # If we find two objects of the same class that are almost identical, 
        # we discard the smaller/subsequent one.
        for j in range(i + 1, len(pred_list)):
            small = pred_list[j]
            if l_label == small['label']:
                inter = np.logical_and(l_mask, small['mask']).sum()
                union = np.logical_or(l_mask, small['mask']).sum()
                iou = inter / union
                
                if iou > iou_threshold:
                    # They represent the same entity. Mark the second one (j) for removal.
                    to_remove.add(j)
                    any_change = True
                    if debug:
                        print(f"   [DEDUPLICATE] Removing duplicate of '{l_label}' (IoU: {iou:.2f})")

        # --- 2. GROUP CLEANING (Explanation Filter) ---
        # Skip if the current large mask is already marked for removal.
        if i in to_remove: continue
        
        # This map tracks which pixels of the 'large' mask are covered by other objects.
        covered_map = np.zeros_like(l_mask, dtype=bool)
        significant_parts_found = True
        
        for j in range(len(pred_list)):
            if i == j or j in to_remove: continue
            
            s_mask = pred_list[j]['mask']
            s_area = s_mask.sum()
            
            # Ignore "noise" or tiny fragments (must be at least 10% of the large mask's size)
            if s_area < (l_area * 0.1): continue
            
            inter_mask = np.logical_and(l_mask, s_mask)
            inter_area = inter_mask.sum()
            
            # Check if the smaller object is "inside" the large one (85% containment)
            if inter_area > (s_area * 0.85):
                covered_map = np.logical_or(covered_map, inter_mask)
                significant_parts_found = True

        # --- FINAL DECISION: Should we discard the large mask? ---
        # We discard it if:
        # A) Most of its area is already covered by smaller parts (explanation_ratio).
        # B) There is evidence of at least one significant sub-part or several fragments.
        explanation_ratio = covered_map.sum() / l_area
        if explanation_ratio > expansion_threshold and significant_parts_found:
            to_remove.add(i)
            any_change = True
            if debug:
                print(f"   [CLEAN] Removing group mask '{l_label}' (Ratio: {explanation_ratio:.2f}, Parts: {significant_parts_found})") 

    # Reconstruct the list excluding the marked indices
    new_pred_list = [p for i, p in enumerate(pred_list) if i not in to_remove]
    return new_pred_list, any_change

class TAOSMetricTracker:
    def __init__(self, iou_threshold=0.4, semantic_threshold=0.85):
        self.iou_threshold = iou_threshold
        self.semantic_threshold = semantic_threshold
        
        # --- Spatial Accumulators ---
        self.image_mious = []
        self.all_instance_ious = [] 
        self.total_gt_count = 0
        self.total_pred_count = 0
        self.total_assignments = 0
        self.total_hallucinations = 0
        
        # For Global Pixel-wise IoU
        self.global_inter_sum = 0
        self.global_union_sum = 0

        # --- Tier-Specific Accumulators ---
        # We track both per-image averages (for Macro) and per-instance (for Micro)
        self.tiers = ['explicit', 'implicit', 'candidate']
        self.tier_image_mious = {t: [] for t in self.tiers}
        self.tier_instance_ious = {t: [] for t in self.tiers}

        # --- Semantic Accumulators ---
        self.all_semantic_scores = []  # List of all cosine similarities for Micro-ASS
        self.image_sc_ratios = []      # Per-image SC ratios for Macro-SC
        self.total_semantic_matches = 0 # Total objects evaluated semantically

    def update(self, metrics, assignments, hallucinations, gt_list, pred_list, shape, semantic_results=None, debug=True):
        """
        metrics: dict of {gt_id: {"iou": float, "state": str}}
        assignments: the dict with metadata and gt_names
        semantic_results: dict returned by SemanticEvaluator.compute_sc_metrics
        debug: if True, prints frame-level results to console
        """
        h, w = shape

        # 1. AMBIGUOUS FILTERING
        valid_gt_list = [gt for gt in gt_list if gt.get('state') != 'ambiguous']
        preds_matching_ambiguous = sum(
            len(v['metadata']) for v in assignments.values() if v.get('state') == 'ambiguous'
        )

        num_gt = len(valid_gt_list)
        if pred_list == None:
            num_pred = 0
        else:
            num_pred = len(pred_list) - preds_matching_ambiguous

        self.total_gt_count += num_gt
        self.total_pred_count += num_pred
        self.total_hallucinations += len(hallucinations)

        # 2. INSTANCE & TIER SPATIAL METRICS
        current_frame_ious = []
        frame_tier_ious = {t: [] for t in self.tiers}
        frame_assignments = 0

        for gt in valid_gt_list:
            gt_id = gt['id']
            state = gt.get('state', 'unknown')
            
            gt_data = metrics.get(gt_id, {"iou": 0.0})            
            iou_val = gt_data["iou"] if isinstance(gt_data, dict) else gt_data
            
            current_frame_ious.append(iou_val)
            self.all_instance_ious.append(iou_val)
            
            if iou_val > self.iou_threshold:
                self.total_assignments += 1
                frame_assignments += 1
            
            if state in self.tiers:
                self.tier_instance_ious[state].append(iou_val)
                frame_tier_ious[state].append(iou_val)

        # Update Overall Macro mIoU
        if valid_gt_list:
            self.image_mious.append(np.mean(current_frame_ious))

        # Update Tier Macro mIoU
        for t in self.tiers:
            if frame_tier_ious[t]:
                self.tier_image_mious[t].append(np.mean(frame_tier_ious[t]))

        # 3. GLOBAL PIXEL-WISE IOU
        all_gt_mask = np.zeros((h, w), dtype=bool)
        for gt in valid_gt_list: all_gt_mask |= gt['mask']
        
        all_pred_mask = np.zeros((h, w), dtype=bool)
        for gt_id, data in assignments.items():
            if data.get('state') != 'ambiguous':
                all_pred_mask |= data['mask']
        for hall in hallucinations:
            all_pred_mask |= hall['mask']
        
        frame_inter = np.logical_and(all_gt_mask, all_pred_mask).sum()
        frame_union = np.logical_or(all_gt_mask, all_pred_mask).sum()
        self.global_inter_sum += frame_inter
        self.global_union_sum += frame_union

        # 4. SEMANTIC METRICS ACCUMULATION
        if semantic_results and semantic_results.get("matched_count", 0) > 0:
            self.image_sc_ratios.append(semantic_results["sc"])
            if "raw_similarities" in semantic_results:
                self.all_semantic_scores.extend(semantic_results["raw_similarities"])
                self.total_semantic_matches += semantic_results["matched_count"]

        # --- 5. DEBUG PRINT BLOCK ---
        if debug:
            f_prec = frame_assignments / num_pred if num_pred > 0 else 0
            f_rec = frame_assignments / num_gt if num_gt > 0 else 0
            f_px_iou = frame_inter / frame_union if frame_union > 0 else 0
            
            print(f"\n>>> [FRAME DEBUG]")
            print(f"    Spatial:  mIoU={np.mean(current_frame_ious):.4f} | Px-IoU={f_px_iou:.4f}" if valid_gt_list else "    Spatial:  N/A (No valid GT)")
            
            # Print breakdown by tier
            tier_str = " | ".join([f"{t[:3].upper()}:{np.mean(frame_tier_ious[t]):.2f}" for t in self.tiers if frame_tier_ious[t]])
            if tier_str: print(f"    Tiers:    {tier_str}")
            
            print(f"    Detect:   P={f_prec:.2f} | R={f_rec:.2f} | FP={len(hallucinations)}")
            
            if semantic_results and semantic_results.get("matched_count", 0) > 0:
                print(f"    Semantic: SC={semantic_results['sc']:.4f} | ASS={semantic_results.get('ass', 0):.4f}")
            print(f"-------------------------------------------------------")

    def report(self):
        # --- Spatial Calculations ---
        precision = self.total_assignments / self.total_pred_count if self.total_pred_count > 0 else 0
        recall = self.total_assignments / self.total_gt_count if self.total_gt_count > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        macro_miou = np.mean(self.image_mious) if self.image_mious else 0
        micro_miou = np.mean(self.all_instance_ious) if self.all_instance_ious else 0
        global_pixel_iou = self.global_inter_sum / self.global_union_sum if self.global_union_sum > 0 else 0
        
        hallucination_rate = (self.total_hallucinations / self.total_pred_count) * 100 if self.total_pred_count > 0 else 0
        # Note: self.total_pred_count already excludes ambiguous matches
        fp_per_image = self.total_hallucinations / len(self.image_mious) if self.image_mious else 0

        # --- Semantic Calculations ---
        # Macro SC: Average of per-image ratios
        macro_sc = np.mean(self.image_sc_ratios) if self.image_sc_ratios else 0
        
        # Micro SC and ASS: Calculated from the pool of all matched objects
        if self.all_semantic_scores:
            scores_array = np.array(self.all_semantic_scores)
            micro_ass = np.mean(scores_array)
            micro_sc = np.mean(scores_array >= self.semantic_threshold)
        else:
            micro_ass = 0
            micro_sc = 0

        # --- TA-mIoU Calculation (Weighted Tier Average) ---
        weights = {'explicit': 1.0, 'implicit': 0.8, 'candidate': 0.4}
        ta_num, ta_den = 0, 0
        for t in self.tiers:
            ma = np.mean(self.tier_image_mious[t]) if self.tier_image_mious[t] else 0
            ta_num += ma * weights[t]
            ta_den += weights[t]
        ta_miou = ta_num / ta_den if ta_den > 0 else 0


        print("\n" + "="*50)
        print(f"{'TAOS EVALUATION REPORT':^50}")
        print("="*50)
        print(f"{'OVERALL SPATIAL PERFORMANCE':<30}")
        print(f"  Global Pixel-wise IoU:    {global_pixel_iou:.4f}")
        print(f"  mIoU (Macro/Image):        {macro_miou:.4f}")
        print(f"  mIoU (Micro/Object):       {micro_miou:.4f}")
        print(f"  TA-mIoU (Task-Aware):      {ta_miou:.4f}") 
        print("-" * 50)
        print(f"{'DETECTION METRICS':<30}")
        print(f"  Recall:                    {recall:.4f}")
        print(f"  Precision:                 {precision:.4f}")
        print(f"  F1-Score:                  {f1:.4f}")
        print("-" * 50)
        print(f"{'TIER-WISE mIoU BREAKDOWN':<35} {'Macro':<10} | {'Micro':<10}")
        print("-" * 50)
        for t in self.tiers:
            ma = np.mean(self.tier_image_mious[t]) if self.tier_image_mious[t] else 0
            mi = np.mean(self.tier_instance_ious[t]) if self.tier_instance_ious[t] else 0
            print(f"  {t.capitalize():<28} {ma:.4f}     | {mi:.4f}")
        print(f"{'SEMANTIC REASONING (Strict)':<30}")
        print(f"  Semantic Consistency (Macro): {macro_sc:.4f}")
        print(f"  Semantic Consistency (Micro): {micro_sc:.4f}")
        print(f"  Avg Semantic Similarity (ASS): {micro_ass:.4f}")
        print(f"  Total Semantic Matches:    {self.total_semantic_matches}")
        print("-" * 50)
        print(f"{'ROBUSTNESS':<30}")
        print(f"  Hallucination Rate:        {hallucination_rate:.2f}%")
        print(f"  FPs per Image:             {fp_per_image:.2f}")
        print("="*50)

    def save_report_image(self, filepath):
        """
        Generates and saves the full TAOS (Image-level) report to a text file 
        and prints LaTeX rows with placeholders for video metrics.
        """
        import numpy as np

        # --- 1. SPATIAL CALCULATIONS ---
        precision = self.total_assignments / self.total_pred_count if self.total_pred_count > 0 else 0
        recall = self.total_assignments / self.total_gt_count if self.total_gt_count > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        macro_miou = np.mean(self.image_mious) if self.image_mious else 0
        micro_miou = np.mean(self.all_instance_ious) if self.all_instance_ious else 0
        global_pixel_iou = self.global_inter_sum / self.global_union_sum if self.global_union_sum > 0 else 0
        
        thr = (self.total_hallucinations / self.total_pred_count) * 100 if self.total_pred_count > 0 else 0
        fp_img = self.total_hallucinations / len(self.image_mious) if self.image_mious else 0

        # --- 2. SEMANTIC CALCULATIONS ---
        macro_sc = np.mean(self.image_sc_ratios) if self.image_sc_ratios else 0
        
        if self.all_semantic_scores:
            scores_array = np.array(self.all_semantic_scores)
            micro_ass = np.mean(scores_array)
            micro_sc = np.mean(scores_array >= self.semantic_threshold)
        else:
            micro_ass = 0
            micro_sc = 0

        # --- 3. TIER-WISE & TA-mIoU (Weighted) ---
        # Weights: Explicit (1.0), Implicit (0.8), Candidate (0.4)
        weights = {'explicit': 1.0, 'implicit': 0.8, 'candidate': 0.4}
        ta_num, ta_den = 0, 0
        tier_stats = {}
        
        for t in self.tiers:
            ma = np.mean(self.tier_image_mious[t]) if self.tier_image_mious[t] else 0
            mi = np.mean(self.tier_instance_ious[t]) if self.tier_instance_ious[t] else 0
            tier_stats[t] = {'macro': ma, 'micro': mi}
            
            ta_num += ma * weights.get(t, 0)
            ta_den += weights.get(t, 0)
            
        ta_miou = ta_num / ta_den if ta_den > 0 else 0

        # --- 4. WRITE TO .TXT FILE ---
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(f"{'TAOS IMAGE BENCHMARK REPORT':^60}\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Global Pixel IoU: {global_pixel_iou:.4f}\n")
            f.write(f"mIoU (Macro):      {macro_miou:.4f}\n")
            f.write(f"mIoU (Micro):      {micro_miou:.4f}\n")
            f.write(f"TA-mIoU (Image):   {ta_miou:.4f}\n\n")

            f.write("-" * 60 + "\n")
            f.write(f"{'TIER BREAKDOWN':<25} {'mIoU (Macro)':<15} | {'mIoU (Micro)':<15}\n")
            f.write("-" * 60 + "\n")
            for t in self.tiers:
                f.write(f"  {t.capitalize():<23} {tier_stats[t]['macro']:.4f}         | {tier_stats[t]['micro']:.4f}\n")
            f.write("\n")

            f.write(f"{'SEMANTIC REASONING':<40}\n")
            f.write(f"  ASS (Similarity): {micro_ass:.4f}\n")
            f.write(f"  SC (Macro):       {macro_sc:.4f}\n")
            f.write(f"  SC (Micro):       {micro_sc:.4f}\n\n")

            f.write(f"{'DETECTION & ROBUSTNESS':<40}\n")
            f.write(f"  Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}\n")
            f.write(f"  THR: {thr:.2f}% | FP/img: {fp_img:.3f}\n")
            f.write("="*60 + "\n")

        # --- 5. GENERATE LATEX ROWS (With Video Placeholders) ---
        print("\n" + "#"*20 + " COPY FOR LATEX (IMAGE ONLY) " + "#"*20)
        
        # TABLE 1: Main Results
        # G-IoU & mIoU & mExp & mImp & mCand & TA-m & JExp & JImp & JCand & JTot
        # Nota: J stats se ponen como --
        m_exp = tier_stats.get('explicit', {}).get('macro', 0)
        m_imp = tier_stats.get('implicit', {}).get('macro', 0)
        m_can = tier_stats.get('candidate', {}).get('macro', 0)
        
        row1 = (f"{global_pixel_iou:.3f} & {macro_miou:.3f} & {m_exp:.3f} & {m_imp:.3f} & "
                f"{m_can:.3f} & {ta_miou:.3f} & -- & -- & -- & -- \\\\")
        
        print("\nTABLE 1 (Main Results - Image Metrics + Video Placeholders):")
        print(row1)

        # TABLE 2: Diagnostics
        # Precision & Recall & F1 & THR & FP/img & SC (Macro)
        row2 = f"{precision:.3f} & {recall:.3f} & {f1:.3f} & {thr:.2f} & {fp_img:.3f} & {macro_sc:.3f} \\\\"
        
        print("\nTABLE 2 (Diagnostics):")
        print(row2)
        print("#"*65 + "\n")

        return True

class SemanticEvaluator:
    def __init__(self, model_name="google/siglip-base-patch16-224", threshold=0.85):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading SigLIP ({model_name}) for semantic evaluation...")
        
        processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = processor.tokenizer
        self.model = SiglipModel.from_pretrained(model_name).to(self.device).eval()
        
        self.threshold = threshold

    @torch.no_grad()
    def compute_sc_metrics(self, assignments, debug=False):
        """
        Computes semantic metrics for provided assignments, ignoring ambiguous objects.
        assignments: dict containing 'metadata', 'gt_name', and 'state'.
        debug: if True, prints a detailed table of matches and scores.
        """
        if not assignments:
            return {"sc": 0.0, "ass": 0.0, "matched_count": 0, "raw_similarities": []}

        gt_texts = []
        pred_texts = []
        object_ids = []

        # 1. Filtering and data preparation
        for obj_id, v in assignments.items():
            try:
                # CRITICAL CHANGE: Ignore ambiguous objects to stay in sync with TAOSMetricTracker
                if v.get('state') == 'ambiguous':
                    continue

                # Extracting labels
                pred_label = v['metadata'][0].get('label', "").strip().lower()
                gt_label = v.get('gt_name', "").strip().lower()

                # Only evaluate if a spatial match exists (both labels must be present)
                if pred_label == "" or gt_label == "":
                    continue

                gt_texts.append(gt_label)
                pred_texts.append(pred_label)
                object_ids.append(obj_id)
                
            except (KeyError, IndexError, TypeError):
                continue

        if not gt_texts:
            if debug: 
                print(">>> [SEMANTIC DEBUG] No valid spatial matches found for evaluation.")
            return {"sc": 0.0, "ass": 0.0, "matched_count": 0, "raw_similarities": []}

        # 2. Batch Inference
        inputs_gt = self.tokenizer(gt_texts, padding=True, return_tensors="pt").to(self.device)
        inputs_pred = self.tokenizer(pred_texts, padding=True, return_tensors="pt").to(self.device)

        feat_gt = self.model.get_text_features(**inputs_gt)[1]
        feat_pred = self.model.get_text_features(**inputs_pred)[1]

        # L2 Normalization for Cosine Similarity
        feat_gt /= feat_gt.norm(dim=-1, keepdim=True)
        feat_pred /= feat_pred.norm(dim=-1, keepdim=True)

        # Compute cosine similarities
        similarities = (feat_gt * feat_pred).sum(dim=-1).cpu().tolist()

        # 3. Debug Output
        if debug:
            print(f"\n{'='*85}")
            print(f"{'OBJECT ID':<15} | {'GT LABEL':<22} | {'PRED LABEL':<22} | {'SCORE':<7} | {'MATCH'}")
            print(f"{'-'*85}")
            for i, score in enumerate(similarities):
                # Hard identity check: if strings are identical, report 1.0
                final_score = 1.0 if gt_texts[i] == pred_texts[i] else score
                
                is_match = "✅" if final_score >= self.threshold else "❌"
                print(f"{object_ids[i]:<15} | {gt_texts[i]:<22} | {pred_texts[i]:<22} | {final_score:.4f} | {is_match}")
            print(f"{'='*85}\n")

        # 4. Final results calculation
        sim_tensor = torch.tensor(similarities)
        
        # We re-apply the threshold check on the tensor
        sc_ratio = (sim_tensor >= self.threshold).float().mean().item()
        ass = sim_tensor.mean().item()

        return {
            "sc": sc_ratio,
            "ass": ass,
            "matched_count": len(gt_texts),
            "raw_similarities": similarities 
        }
# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

def get_color(idx):
    """
    Generates distinct, vibrant colors using the HSV color space.
    This ensures that consecutive objects in a list receive high-contrast colors.
    """
    # 1. Apply an offset to skip the very first few base colors
    idx = idx + 20
    
    # 2. Calculate Hue. In OpenCV's 8-bit HSV implementation, Hue ranges from 0-179.
    # Multiplying by 37 (a prime number) creates a "jump" effect on the color wheel.
    # This prevents consecutive indices from having similar colors (like two shades of blue).
    hue = (idx * 37) % 180 
    
    # 3. Create a 1x1 HSV pixel with maximum Saturation and Value (Brightness)
    # Saturation=255 and Value=255 ensure the color is as vibrant as possible.
    hsv_color = np.uint8([[[hue, 255, 255]]])
    
    # 4. Convert the HSV pixel to BGR format for compatibility with OpenCV
    # OpenCV's cvtColor expects a specific array shape, so we extract the [0][0] pixel.
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    
    # 5. Return the result as a standard BGR tuple of integers
    return tuple(int(c) for c in bgr_color)

def put_text_outline(img, text, pos, font, scale, color, thickness):
    """
    Draws text with a black outline to ensure maximum readability against any background.
    """
    x, y = pos
    
    # 1. Draw the outline (Halo effect)
    # We use a black color (0,0,0) and a slightly larger thickness (thickness + 2).
    # This creates a dark border around where the main text will be placed.
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    
    # 2. Draw the main text
    # We place the colored text directly over the black outline.
    # LINE_AA is used for anti-aliased lines, making the text look smoother.
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

def draw_predictions_with_boxes(img, pred_assignments, hallucinations, alpha=0.4):
    """
    Visualizes model predictions on the image, distinguishing between matched 
    objects (True Positives) and hallucinations (False Positives).
    
    It highlights the model's actual reasoning by displaying Qwen's predicted labels 
    rather than the Ground Truth categories.
    """
    canvas = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # --- 1. DRAW ASSIGNMENTS (Matched Predictions / True Positives) ---
    # These are predictions that successfully overlapped with a GT entity.
    for i, (gt_id, data) in enumerate(pred_assignments.items()):
        # Convert boolean mask to uint8 for OpenCV operations
        mask = data["mask"].astype(np.uint8)
        color = get_color(i)
        
        # A. Apply Shading (Alpha Blending)
        # This creates the semi-transparent "fill" effect inside the mask.
        canvas[mask > 0] = canvas[mask > 0] * (1 - alpha) + np.array(color) * alpha
        
        # B. Draw Solid Outlines (Contours)
        # This provides sharp edges to the aggregated masks (Many-to-One).
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)
        
        # C. Label Logic: Extract Qwen's original predictions
        # Since one GT might be explained by multiple prediction fragments, 
        # we collect all unique labels provided by the model for this cluster.
        qwen_labels = list(set([m['label'] for m in data.get('metadata', [])]))
        label_text = ", ".join(qwen_labels) if qwen_labels else "unknown"
        
        # D. Bounding Box: Calculated dynamically from the mask's extent
        y, x = np.where(mask > 0)
        if len(x) > 0:
            x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            # Display "P: [Qwen Label]" to show the model's classification
            cv2.putText(canvas, f"P: {label_text}", (x1, y1-10), font, 0.6, color, 2)

    # --- 2. DRAW HALLUCINATIONS (False Positives) ---
    # These are predictions that did not match any GT object (thresholded by IoP).
    hall_alpha = 0.25 # Lighter shade for hallucinations
    for hall in hallucinations:
        mask = hall["mask"].astype(np.uint8)
        color = (0, 0, 255) # Always Red for hallucinations
        
        # A. Apply Red Shading
        canvas[mask > 0] = canvas[mask > 0] * (1 - hall_alpha) + np.array(color) * hall_alpha
        
        # B. Draw Solid Red Outline
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2)
        
        # C. Extract the label Qwen assigned to this "phantom" object
        qwen_name = hall.get('label', 'hallucination')
        
        # D. Bounding Box and FP Tag
        y, x = np.where(mask > 0)
        if len(x) > 0:
            x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            # Clearly mark as False Positive while showing the model's intent
            cv2.putText(canvas, f"FP: {qwen_name}", (x1, y1-10), font, 0.6, color, 2)
        
    return canvas

def draw_pred_with_boxes(img, pred_list, alpha=0.4):
    """
    Renders Ground Truth masks on the image with alpha-blended shading, 
    solid external contours, and bounding boxes.
    """
    canvas = img.copy()
    
    for i, pred in enumerate(pred_list):
        # 1. Retrieve the mask using flexible key mapping
        # We check for both 'mask' and 'mask_solid' to support the Dual GT logic.
        mask = pred.get('mask', pred.get('mask_solid', None))
        if mask is None: continue
        
        ## DEBUG: to show only certain masks
        # if pred['label'] != 'pot' and pred['label'] != 'pan':
        #     continue

        # Convert boolean or generic mask to uint8 for OpenCV compatibility
        mask = mask.astype(np.uint8)
        color = get_color(i)
        
        # 2. Draw Shading (Alpha Blending)
        # We apply the color only where the mask pixels are active (> 0).
        canvas[mask > 0] = canvas[mask > 0] * (1 - alpha) + np.array(color) * alpha
        
        # 3. DRAW SOLID EXTERNAL CONTOUR
        # RETR_EXTERNAL retrieves only the outermost outline, which is perfect 
        # for representing "solid" objects even if the internal mask has gaps.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2) # Thickness 2 for a bold look
        
        # 4. Draw BBox and Label
        # We derive the bounding box coordinates directly from the active mask pixels.
        y, x = np.where(mask)
        if len(x) > 0:
            x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)
            
            # Draw the rectangular enclosure
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            
            # Place the pred label text above the box
            cv2.putText(canvas, f"Pred: {pred.get('label', 'obj')}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
    return canvas

def draw_gt_with_boxes(img, gt_list, alpha=0.4):
    """
    Renders Ground Truth masks on the image with alpha-blended shading, 
    solid external contours, and bounding boxes.
    """
    canvas = img.copy()
    
    for i, gt in enumerate(gt_list):
        # 1. Retrieve the mask using flexible key mapping
        # We check for both 'mask' and 'mask_solid' to support the Dual GT logic.
        mask = gt.get('mask', gt.get('mask_solid', None))
        if mask is None: continue
        
        # Convert boolean or generic mask to uint8 for OpenCV compatibility
        mask = mask.astype(np.uint8)
        color = get_color(i)
        
        # 2. Draw Shading (Alpha Blending)
        # We apply the color only where the mask pixels are active (> 0).
        canvas[mask > 0] = canvas[mask > 0] * (1 - alpha) + np.array(color) * alpha
        
        # 3. DRAW SOLID EXTERNAL CONTOUR
        # RETR_EXTERNAL retrieves only the outermost outline, which is perfect 
        # for representing "solid" objects even if the internal mask has gaps.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2) # Thickness 2 for a bold look
        
        # 4. Draw BBox and Label
        # We derive the bounding box coordinates directly from the active mask pixels.
        y, x = np.where(mask)
        if len(x) > 0:
            x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)
            
            # Draw the rectangular enclosure
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            
            # Place the GT label text above the box
            cv2.putText(canvas, f"GT: {gt.get('name', 'obj')}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
    return canvas

def draw_tiers_gt(img_bgr, gt_list):
    """
    Colors masks based on their functional tier.
    """
    canvas = img_bgr.copy()
    # Define tier colors (BGR)
    tier_colors = {
        'explicit': (0, 255, 0),    # Green
        'implicit': (255, 100, 0),  # Blue
        'candidate': (0, 165, 255), # Orange
        'ambiguous': (0, 0, 255) # Red
    }
    
    for gt in gt_list:
        mask = gt['mask'].astype(bool)
        state = gt.get('state', 'unknown').lower()
        color = tier_colors.get(state, (255, 255, 255)) # Default White
        
        # Overlay color with 40% transparency
        canvas[mask] = canvas[mask] * 0.6 + np.array(color) * 0.4
        
        # Draw small label near the object
        y, x = np.where(mask)
        if len(x) > 0:
            cv2.putText(canvas, state.upper(), (int(np.min(x)), int(np.min(y)) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
    return canvas

def create_instance_error_map(shape, assignments, hallucinations):
    """
    Generates a pixel-wise diagnostic error map for the entire frame.
    Color Coding:
    - GREEN: True Positives (Overlap between GT and Prediction)
    - BLUE:  False Negatives (Missing parts of the GT)
    - RED:   False Positives (Extra predicted pixels or hallucinations)
    """
    h, w = shape
    # Initialize a black image for the visualization
    error_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Initialize global boolean masks to aggregate errors across all instances
    tp_total = np.zeros((h, w), dtype=bool)
    fp_total = np.zeros((h, w), dtype=bool)
    fn_total = np.zeros((h, w), dtype=bool)
    
    for gt_id, data in assignments.items():

        # Skip objects marked as ambiguous so they don't 'pollute' the diagnostic map
        if data.get('state') == 'ambiguous':
            continue

        # CORE TRICK: Use the 'best_gt_mask' (either hollow or solid) 
        # that was selected during the Max-IoU assignment phase.
        gt_mask = data["best_gt_mask"].astype(bool) 
        pred_mask = data["mask"].astype(bool)
        
        # 1. True Positives: Intersection of GT and Prediction
        tp_total |= (gt_mask & pred_mask)
        
        # 2. False Negatives: GT pixels that were NOT predicted
        fn_total |= (gt_mask & ~pred_mask)
        
        # 3. False Positives: Predicted pixels that are NOT in the GT
        fp_total |= (~gt_mask & pred_mask)

    # 4. Handle Hallucinations:
    # Any prediction that didn't match a GT object is added to the False Positive mask.
    for hall in hallucinations:
        fp_total |= hall["mask"].astype(bool)

    # 5. Final Color Assignment (BGR format):
    # Note: Assignments are done in this order so that TPs (Green) appear 
    # "on top" of errors in the final visualization if masks overlap.
    error_img[fn_total] = [255, 0, 0]   # Blue (False Negatives - Missing)
    error_img[fp_total] = [0, 0, 255]   # Red  (False Positives - Extra/Wrong)
    error_img[tp_total] = [0, 255, 0]   # Green (True Positives - Correct)
    
    return error_img

def show_comparison(img_bgr, gt_list, gt_entities, pred_assignments, hallucinations, metrics, task_name, save = False, folder=None, filename = None, stop_per_frame = True):
    """
    Creates a 2x2 panel visualization: 
    Top-Left: GT (Names)      | Top-Right: Preds (Names)
    Bottom-Left: GT (Tiers)   | Bottom-Right: Error Map + Tier Metrics (Black background)
    """
    h, w = img_bgr.shape[:2]
    
    # --- 1. GENERATE THE 4 INDIVIDUAL PANELS ---
    # Top Row
    viz_gt_names = draw_gt_with_boxes(img_bgr, gt_list)
    viz_pred = draw_predictions_with_boxes(img_bgr, pred_assignments, hallucinations)
    
    # Bottom Row
    viz_gt_tiers = draw_tiers_gt(img_bgr, gt_list) # Uses the helper provided before
    viz_err = create_instance_error_map((h, w), pred_assignments, hallucinations)
    
    # --- 2. OVERLAY METRICS ON THE ERROR MAP (Panel 4) ---
    # We draw this BEFORE stacking to make coordinate management easier
    font_scale = 1.2 # Larger font as requested
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    if save:
        output_image_dir = os.path.join(folder,'imgs')
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)
    
    for i, (gt_id, info) in enumerate(metrics.items()):
        # FIX: Extract data from the state-aware dictionary
        if isinstance(info, dict):
            iou_val = info.get('iou', 0.0)
            state = info.get('state', 'unknown')
        else:
            iou_val = info
            state = 'unknown'

        color = get_color(i)
        # Position text inside the 4th panel (viz_err)
        y_pos = 50 + (i * 60) 
        
        # Determine the friendly name
        friendly_name = gt_id
        if gt_id in pred_assignments:
            friendly_name = pred_assignments[gt_id].get("gt_name", gt_id)
        
        # Clean technical prefixes (gt_0_pan -> pan)
        if "_" in friendly_name and friendly_name.startswith("gt_"):
            parts = friendly_name.split("_")
            if len(parts) > 2: friendly_name = "_".join(parts[2:])

        # Format: [STATE] Name: IoU
        text = f"[{state[:3].upper()}] {friendly_name}: {iou_val:.2f}"
        
        # Draw on viz_err (Black background ensures readability)
        cv2.putText(viz_err, text, (20, y_pos), font, font_scale, color, thickness)

    # --- 3. ASSEMBLE 2x2 GRID ---
    top_row = np.hstack((viz_gt_names, viz_pred))
    bottom_row = np.hstack((viz_gt_tiers, viz_err))
    grid_canvas = np.vstack((top_row, bottom_row))
    
    # --- 4. DYNAMIC HEADER LOGIC ---
    header_font_scale = 1.2
    header_margin = 40
    line_height = int(60 * header_font_scale)
    
    max_text_width = grid_canvas.shape[1] - (header_margin * 2)
    words = f"TASK: {task_name}".split(' ')
    lines, current_line = [], ""

    for word in words:
        test_line = current_line + word + " "
        (w_text, h_text), _ = cv2.getTextSize(test_line, font, header_font_scale, thickness)
        if w_text < max_text_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + " "
    lines.append(current_line)

    header_h = (len(lines) * line_height) + (header_margin * 2)
    header = np.zeros((header_h, grid_canvas.shape[1], 3), dtype=np.uint8)

    for i, line in enumerate(lines):
        y_p = header_margin + (i * line_height) + 30
        cv2.putText(header, line.strip(), (header_margin, y_p), font, header_font_scale, (255, 255, 255), thickness)

    # --- 5. FINAL ASSEMBLY ---
    canvas = np.vstack((header, grid_canvas))

    # --- 6. ADD FOOTER LEGEND ---
    # Legend for tiers and error colors
    legend_tier = "TIERS: GREEN=EXP | BLUE=IMP | ORANGE=CAN | RED=AMB"
    legend_err = "ERRORS: GREEN=TP | RED=FP | BLUE=FN"
    
    cv2.putText(canvas, legend_tier, (20, canvas.shape[0] - 50), font, 0.8, (200, 200, 200), 2)
    cv2.putText(canvas, legend_err, (20, canvas.shape[0] - 15), font, 0.8, (200, 200, 200), 2)

    # --- 7. WINDOW MANAGEMENT ---
    

    if save:
        cv2.imwrite(os.path.join(output_image_dir,filename+'.jpg'), canvas)
        return True, canvas
    else:
        win_name = "TAOS Diagnostic - 2x2 View"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, canvas)
        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == 32 or key == 27: return True, canvas # Space or Esc to continue
            if key == ord('q') or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyWindow(win_name)
                return False, canvas
            
            if not stop_per_frame:
                return True, canvas

def show_comparison_individual(img_bgr, gt_list, gt_entities, pred_assignments, hallucinations, metrics, task_name, save=False, folder=None, filename=None, stop_per_frame=True):
    """
    Visualizes results in 4 separate windows/files with high-quality rendering:
    1. GT Grounding (Clean labels, no 'GT:' prefix)
    2. Prediction Grounding (Clean labels, no 'P:' prefix, includes Hallucinations)
    3. Functional Tiers (Contours + shading, NO text labels, Pink for ambiguous)
    4. Diagnostic Error Map (Error legend + metrics list)
    """
    h, w = img_bgr.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    alpha = 0.4
    
    # --- HELPER: CLEAN NAMES (gt_0_pan -> pan) ---
    def clean_label(name):
        # Remove common technical prefixes
        for prefix in ["gt_", "p_", "P: ", "GT: "]:
            if name.startswith(prefix):
                name = name.replace(prefix, "")
        if "_" in name:
            parts = name.split("_")
            # If it follows gt_X_name, take only the name
            if len(parts) > 2 and parts[0].isdigit() or (parts[0] == 'gt' and parts[1].isdigit()):
                return "_".join(parts[2:])
        return name

    # --- PANEL 1: GROUNDING GT ---
    viz_gt = img_bgr.copy()
    for i, gt in enumerate(gt_list):
        if gt.get('state') == 'ambiguous': continue
        mask = gt.get('mask').astype(np.uint8)
        color = get_color(i)
        
        # Shading + Contours
        viz_gt[mask > 0] = viz_gt[mask > 0] * (1 - alpha) + np.array(color) * alpha
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(viz_gt, contours, -1, color, 2)
        
        # Box + Label (Larger font, no prefix)
        y, x = np.where(mask > 0)
        if len(x) > 0:
            x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)
            label = clean_label(gt.get('id', 'obj'))
            cv2.rectangle(viz_gt, (x1, y1), (x2, y2), color, 2)
            cv2.putText(viz_gt, label, (x1, y1-12), font, 1.0, color, 2)

    # --- PANEL 2: GROUNDING PREDICTIONS ---
    viz_pred = img_bgr.copy()
    # Matched Predictions
    for i, (gt_id, data) in enumerate(pred_assignments.items()):
        mask = data["mask"].astype(np.uint8)
        color = get_color(i)
        viz_pred[mask > 0] = viz_pred[mask > 0] * (1 - alpha) + np.array(color) * alpha
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(viz_pred, contours, -1, color, 2)
        
        y, x = np.where(mask > 0)
        if len(x) > 0:
            x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)
            qwen_labels = list(set([m['label'] for m in data.get('metadata', [])]))
            label_text = clean_label(", ".join(qwen_labels))
            cv2.rectangle(viz_pred, (x1, y1), (x2, y2), color, 2)
            cv2.putText(viz_pred, label_text, (x1, y1-12), font, 1.0, color, 2)

    # Hallucinations (Red)
    for hall in hallucinations:
        mask = hall["mask"].astype(np.uint8)
        color = (0, 0, 255)
        viz_pred[mask > 0] = viz_pred[mask > 0] * (1 - 0.25) + np.array(color) * 0.25
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(viz_pred, contours, -1, color, 2)
        y, x = np.where(mask > 0)
        if len(x) > 0:
            x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)
            cv2.rectangle(viz_pred, (x1, y1), (x2, y2), color, 2)
            cv2.putText(viz_pred, f"FP: {clean_label(hall.get('label', 'obj'))}", (x1, y1-12), font, 1.0, color, 2)

    # --- PANEL 3: FUNCTIONAL TIERS (NO TEXT, PINK AMBIGUOUS) ---
    viz_tiers = img_bgr.copy()
    tier_colors = {
        'explicit': (0, 255, 0),    # Green
        'implicit': (255, 100, 0),  # Blue
        'candidate': (0, 165, 255), # Orange
        'ambiguous': (203, 192, 255) # Pink
    }
    for gt in gt_list:
        state = gt.get('state', 'explicit').lower()
        color = tier_colors.get(state, (255, 255, 255))
        mask = gt['mask'].astype(np.uint8)
        # Shading
        viz_tiers[mask > 0] = viz_tiers[mask > 0] * 0.6 + np.array(color) * 0.4
        # Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(viz_tiers, contours, -1, color, 2)

    # --- PANEL 4: ERROR MAP + METRICS ---
    viz_err = create_instance_error_map((h, w), pred_assignments, hallucinations)
    # Overlay Metrics
    for i, (gt_id, info) in enumerate(metrics.items()):
        if isinstance(info, dict):
            iou_val, state = info.get('iou', 0.0), info.get('state', 'unknown')
        else:
            iou_val, state = info, 'unknown'
        y_pos = 50 + (i * 60)
        color = get_color(i)
        label = clean_label(gt_id)
        text = f"[{state[:3].upper()}] {label}: {iou_val:.2f}"
        cv2.putText(viz_err, text, (20, y_pos), font, 1.2, color, 2)
    
    # Legend for Panel 4
    legend_err = "ERRORS: GREEN=TP | RED=FP | BLUE=FN"
    cv2.putText(viz_err, legend_err, (20, h - 25), font, 0.8, (255, 255, 255), 2)

    # --- SAVE OR SHOW ---
    if save:
        out_dir = os.path.join(folder, 'individual_plots')
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(out_dir, f"{base}_gt.jpg"), viz_gt)
        cv2.imwrite(os.path.join(out_dir, f"{base}_pred.jpg"), viz_pred)
        cv2.imwrite(os.path.join(out_dir, f"{base}_tiers.jpg"), viz_tiers)
        cv2.imwrite(os.path.join(out_dir, f"{base}_error.jpg"), viz_err)
        return True, None
    else:
        wins = ["GT Grounding", "Predictions", "Tiers", "Error Analysis"]
        imgs = [viz_gt, viz_pred, viz_tiers, viz_err]
        for title, img in zip(wins, imgs):
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.imshow(title, img)
        
        key = cv2.waitKey(0 if stop_per_frame else 1) & 0xFF
        if key == ord('q'):
            for title in wins: cv2.destroyWindow(title)
            return False, None
        return True, None

def visualize_mask_changes(img_bgr, masks_before, masks_after, title="Refinement"):
    """
    Visualizes the pixel-level differences between two sets of masks.
    Useful for debugging Deduplication, Cleaning, or any refinement steps.
    
    Color Key:
    - BLUE:   Static pixels (existed in both versions).
    - YELLOW: Added pixels (growth or new objects).
    - RED:    Removed pixels (shrinkage or deleted objects).
    """
    h, w = img_bgr.shape[:2]
    # Create a darkened version of the image to serve as a high-contrast background
    diff_viz = (img_bgr * 0.15).astype(np.uint8)
    
    # 1. Map "before" masks by ID for quick O(1) access
    # Uses 'id' if available, otherwise defaults to the list index
    before_map = {m.get('id', i): m['mask'].astype(bool) for i, m in enumerate(masks_before)}
    
    total_added = 0
    total_removed = 0
    processed_ids = set()

    # 2. Compare each "after" object with its corresponding "before" state
    for i, ma in enumerate(masks_after):
        m_id = ma.get('id', i)
        processed_ids.add(m_id)
        mask_after = ma['mask'].astype(bool)
        
        if m_id in before_map:
            mask_before = before_map[m_id]
            
            # Identify pixel-wise changes for this specific object
            added = mask_after & ~mask_before
            removed = mask_before & ~mask_after
            stayed = mask_before & mask_after
            
            # Paint the visualization only if changes occurred to avoid clutter
            if added.any() or removed.any():
                diff_viz[stayed] = [200, 0, 0]   # Blue (Base/Static)
                diff_viz[added] = [0, 255, 255] # Yellow (Growth/Added)
                diff_viz[removed] = [0, 0, 255] # Red (Shrinkage/Removed)
                
                total_added += added.sum()
                total_removed += removed.sum()
        else:
            # Handle entirely new objects appearing in the "after" list
            diff_viz[mask_after] = [0, 255, 255]
            total_added += mask_after.sum()

    # 3. Identify objects that disappeared entirely (Deduplicated or Cleaned)
    for m_id, mask_before in before_map.items():
        if m_id not in processed_ids:
            # Entire mask is marked red as it was removed
            diff_viz[mask_before] = [0, 0, 255] 
            total_removed += mask_before.sum()

    # 4. Console Debugging Output
    print(f"\n--- [DEBUG VIZ: {title}] ---")
    print(f"Total pixels added across all objects: {total_added}")
    print(f"Total pixels removed across all objects: {total_removed}")

    # 5. Build the Triple-Panel Visualization
    # Panels use a dim background to make the colored masks pop
    bg_dim = (img_bgr * 0.3).astype(np.uint8)
    viz_before = draw_pred_with_boxes(bg_dim.copy(), masks_before, alpha=0.5)
    viz_after = draw_pred_with_boxes(bg_dim.copy(), masks_after, alpha=0.5)
    
    # Horizontal stack: [BEFORE] | [CHANGES] | [AFTER]
    canvas = np.hstack((viz_before, diff_viz, viz_after))
    canvas = np.hstack((viz_before, viz_after))

    
    # Add a descriptive header
    header = np.zeros((80, canvas.shape[1], 3), dtype=np.uint8)
    msg = f"{title} | Blue: Static | Yellow: +{total_added}px | Red: -{total_removed}px"
    put_text_outline(header, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    canvas = np.vstack((header, canvas))

    # 6. Window Management
    win_name = f"DEBUG_{title.replace(' ', '_')}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, canvas)
    
    # Event loop: wait for Space/Esc to continue, or 'q' to stop execution
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == 32 or key == 27: # Space or Escape
            break
        if key == ord('q') or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1: 
            cv2.destroyWindow(win_name)
            return False
            
    cv2.destroyWindow(win_name)
    return True

def visualize_topology_comparison(img, mask_solid, mask_hollow, mask_pred, padding=40, alpha_gt=0.5):
    """
    Layout:
    [ IMAGEN COMPLETA ] | [ GT SOLID ] [ GT HOLLOW ]
    [   CON BBOX      ] | [ PRED ON S ] [ PRED ON H ]
    
    Colores: GT Azul/Cian (con Alpha) | PRED Verde Sólido
    """
    # 1. Encontrar BBox para el recorte y el marcado
    combined = (mask_solid | mask_hollow | mask_pred).astype(np.uint8)
    if not combined.any(): return
    x, y, w, h = cv2.boundingRect(combined)
    
    # Coordenadas del recorte con padding
    H, W = img.shape[:2]
    y1, y2 = max(0, y - padding), min(H, y + h + padding)
    x1, x2 = max(0, x - padding), min(W, x + w + padding)
    crop_bg = img[y1:y2, x1:x2].copy()

    def blend_mask(base, mask, color, a):
        out = base.copy()
        m = mask.astype(bool)
        if not m.any(): return out
        roi = out[m].astype(float)
        blended = roi * (1 - a) + np.array(color, dtype=float) * a
        out[m] = blended.astype(np.uint8)
        return out

    def apply_solid_mask(base, mask, color):
        out = base.copy()
        m = mask.astype(bool)
        if m.any():
            out[m] = color
        return out

    def add_title(panel, text):
        # Añade una franja negra superior para el título del cuadrante
        res = panel.copy()
        # cv2.rectangle(res, (0, 0), (res.shape[1], 35), (0, 0, 0), -1)
        cv2.putText(res, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return res

    # 2. Preparar los 4 cuadrantes (recortados)
    c_solid = [255, 0, 0]    # Azul
    c_hollow = [255, 255, 0] # Cian
    c_pred = [0, 255, 0]     # Verde Sólido

    # Fila Superior: GT con Alpha
    q_top_left = blend_mask(crop_bg, mask_solid[y1:y2, x1:x2], c_solid, alpha_gt)
    q_top_right = blend_mask(crop_bg, mask_hollow[y1:y2, x1:x2], c_hollow, alpha_gt)

    # Fila Inferior: Predicción SÓLIDA sobre los GT anteriores
    q_bot_left = apply_solid_mask(q_top_left, mask_pred[y1:y2, x1:x2], c_pred)
    q_bot_right = apply_solid_mask(q_top_right, mask_pred[y1:y2, x1:x2], c_pred)

    # Añadir títulos a cada cuadrante
    q_top_left = add_title(q_top_left, "GT Mask (Solid)")
    q_top_right = add_title(q_top_right, "GT Mask (Hollow)")
    q_bot_left = add_title(q_bot_left, "Prediction Mask over GT Mask (Solid)")
    q_bot_right = add_title(q_bot_right, "Prediction Mask over GT Mask (Hollow)")

    # 3. Crear bloque 2x2
    top_row = np.hstack((q_top_left, q_top_right))
    bot_row = np.hstack((q_bot_left, q_bot_right))
    right_panel = np.vstack((top_row, bot_row))

    # 4. Panel Izquierdo (Imagen Completa)
    left_panel = img.copy()
    cv2.rectangle(left_panel, (x, y), (x + w, y + h), [0, 255, 0], 3)
    
    # Reescalar
    target_h = right_panel.shape[0]
    aspect_ratio = left_panel.shape[1] / left_panel.shape[0]
    target_w = int(target_h * aspect_ratio)
    left_panel_res = cv2.resize(left_panel, (target_w, target_h))
    
    # Añadir título al panel izquierdo
    # left_panel_res = add_title(left_panel_res, "GLOBAL CONTEXT")

    # 5. Final
    final_canvas = np.hstack((left_panel_res, right_panel))

    cv2.namedWindow("TAOS Final Diagnostic", cv2.WINDOW_NORMAL)
    cv2.imshow("TAOS Final Diagnostic", final_canvas)
    cv2.waitKey(0)

# ==========================================
# LOADING DATA AND MAIN LOOP
# ==========================================

def load_pkl_gz(path):

    with gzip.open(path, 'rb') as f:
        return pickle.load(f)

def run_evaluation(gt_dir, pred_dir, img_dir, limit=None, specific_ids=None):
    """
    Main evaluation pipeline. Iterates through the dataset, preprocesses Ground Truth 
    and Predictions, and computes metrics using the Max-IoU / Many-to-One logic.
    """
    # 1. File Discovery
    all_files = sorted(glob.glob(os.path.join(gt_dir, "*.pkl.gz")))
    
    # # Slice the file list for specific debugging ranges
    # eval_files = eval_files[105:]
    # eval_files = [eval_files[2]]

    # Initialize the tracker
    tracker = TAOSMetricTracker(iou_threshold=0.4)
    eval_semantic = SemanticEvaluator()


    # Read the sequences:
    sequence_groups = defaultdict(list)
    for path in all_files:
        filename = os.path.basename(path)
        seq_id = filename.split('--')[0]
        sequence_groups[seq_id].append(path)

    # 2. Frame-by-Frame Processing Loop
    for gt_path in tqdm(all_files, desc="Evaluating"):
        base_name = os.path.basename(gt_path).replace(".pkl.gz", "")
        base_name = os.path.basename(base_name).replace("GT_", "")



        # Load the original RGB image for visualization
        img_path = os.path.join(img_dir, f"{base_name}.jpg")
        image = cv2.imread(img_path)
        if image is None: continue
        
        # Load Ground Truth data from compressed pickle
        gt_data = load_pkl_gz(gt_path)
        
        # --- GROUND TRUTH PREPROCESSING ---
        # Flatten the hierarchical GT dictionary (class -> instances) into a simple list
        gt_list = []
        for name, insts in gt_data.get("confirmed_objects", {}).items():
            for i, inst in enumerate(insts):
                if inst.get("mask") is not None:
                    # Maintain name, unique index, and bounding box
                    gt_list.append({
                        "mask": inst["mask"].astype(bool),
                        "name": name,
                        "id": f"{name}_{i}",
                        "bbox": inst.get("bbox"),
                        "state": inst.get("state")
                    })


        # cv2.namedWindow("debug", cv2.WINDOW_NORMAL)


        # COLLAPSE GT: Merge multiple instances of the same label into a single entity.
        # This ensures that if 3 'knives' exist, they are evaluated as one 'knife' mask.
        debug_COLLAPSE = False
        if debug_COLLAPSE:
            viz_before = draw_gt_with_boxes(image.copy(), gt_list, alpha=0.5)

        gt_list = collapse_gt_by_label(gt_list)

        if debug_COLLAPSE:
            viz_after = draw_gt_with_boxes(image.copy(), gt_list, alpha=0.5)
            canvas = np.hstack((viz_before, viz_after))
            cv2.imshow("debug", canvas)
            cv2.waitKey(0)

        # --- PREDICTION PREPROCESSING ---
        # Locate corresponding prediction file
        pred_path = os.path.join(pred_dir, f"{base_name}.pkl.gz")
        if not os.path.exists(pred_path): continue
        
        pred_data = load_pkl_gz(pred_path)
        
        # Flatten predictions in the same way as GT
        pred_list = []
        for label, insts in pred_data.get("confirmed_objects", {}).items():
            for i, inst in enumerate(insts): # Tracking i for unique pred IDs
                if inst.get("mask") is not None:
                    pred_list.append({
                        "mask": inst["mask"].astype(bool),
                        "label": label,
                        "id": f"pred_{label}_{i}",
                        "bbox": inst.get("bbox")
                    })
        
        # --- IDENTITY SYNCHRONIZATION ---
        # Assign unique IDs to every object to track them through refinement/cleaning steps.
        for i, m in enumerate(gt_list):
            m['id'] = f"gt_{i}_{m.get('name', 'obj')}"

        for i, m in enumerate(pred_list):
            m['id'] = f"pred_{i}_{m.get('label', 'obj')}"



        # --- PREDICTION CLEANING ---
        # Deduplicate overlapping predictions (e.g., SAM producing redundant masks)
        debug_REDUNDANT = False
        pred_list_before_clean = copy.deepcopy(pred_list) 
        pred_list, changed_pred_clean = clean_redundant_predictions(pred_list, debug=debug_REDUNDANT)
        
        # Visualize changes if any redundant masks were removed
        if changed_pred_clean and debug_REDUNDANT:
            visualize_mask_changes(image, pred_list_before_clean, pred_list, title="PRED Cleaning")

        # --- DUAL GT GENERATION ---
        # Generate the 'Hollow' (donut) and 'Solid' (disc) versions of the GT
        # to allow for topological flexibility during evaluation.
        debug_DUAL = False
        dual_gt_list = prepare_dual_gt(gt_list)

        # --- CORE EVALUATION ---
        # Perform Many-to-One matching and calculate Max-IoU for each GT entity.
        metrics, gt_entities, assignments, hallucinations = evaluate_agnostic_frame_max_iou(
            dual_gt_list, 
            pred_list, 
            image.shape[:2]
        )


        if debug_DUAL:
            print(assignments.keys())
            object_name = 'gt_1_mug'
            
            visualize_topology_comparison(image, assignments[object_name]['mask_solid'], assignments[object_name]['mask_hollow'], assignments[object_name]['mask'])

        # SEMANTIC CONSISTENCY
        semantic_res = eval_semantic.compute_sc_metrics(assignments, debug=False)
        # print(f"SC Score: {semantic_res['sc']:.4f}, ASS: {semantic_res['ass']:.4f}")


        # UPDATE TRACKER
        tracker.update(metrics, assignments, hallucinations, gt_list, pred_list, image.shape[:2], semantic_results=semantic_res, debug=False)

        # --- VISUALIZATION ---
        # Show the triple-panel comparison (GT | PRED | ERROR MAP)
        task_name = gt_data.get("task", "No Task Name Found")
        continue_eval, _ = show_comparison(image, gt_list, gt_entities, assignments, hallucinations, metrics, task_name, save = False, folder=pred_dir, filename=base_name, stop_per_frame = False)
        # continue_eval, _ = show_comparison_individual(image, gt_list, gt_entities, assignments, hallucinations, metrics, task_name, save = True, folder=pred_dir, filename=base_name, stop_per_frame = True)

        if not continue_eval:
            # If the user presses 'q' or closes the window, stop the evaluation
            break

    # FINAL REPORT
    tracker.report()
    tracker.save_report_image(os.path.join(pred_dir, "final_metrics_report.txt"))

    # Cleanup OpenCV windows at the end
    cv2.destroyAllWindows()

# # --- EXECUTION BLOCK ---
# if __name__ == "__main__":
#     # Define directory paths for Ground Truth metadata, 
#     # model predictions, and raw source images.
#     GT_DIR =  "./meta_pkl"
#     IMG_DIR = ".//images"
#     PRED_DIR = "./4_qwen2B_sam3"
    
#     # Run the evaluation pipeline.
#     # The 'limit' parameter can be adjusted to process a specific number of frames.
#     # Set limit=None to evaluate the entire dataset.
#     run_evaluation(GT_DIR, PRED_DIR, IMG_DIR, limit=10000)
    






