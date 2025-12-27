import numpy as np
import cv2
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, \
                        compute_dice_coefficient
from tqdm import tqdm
import argparse
from main_utils import load_cfg_from_cfg_file
join = os.path.join
basename = os.path.basename

# Argument parser
def get_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
    "--config-file",
    required=True,
    type=str,
    help="Path to config file",
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help="Random seed for reproducibility."
    )

    parser.add_argument(
        '--prompt_design',
        type=str,
        default="original",
        help="Text prompt design."
    )

    parser.add_argument(
        "--data_percentage", 
        type=int, 
        default=100, 
        help="Percentage of data to use.")
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="", 
        help="output directory")
    
    parser.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="modify config options using the command-line",
        )

    args = parser.parse_args()

    cfg = load_cfg_from_cfg_file(args.config_file)

    cfg.merge_from_list(args.opts)

    cfg.update({k: v for k, v in vars(args).items()})    

    return cfg

cfg = get_arguments()

gt_path = cfg.DATASET.TEST_PATH + 'label'

backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")
results_name = (f"MedCLIPSeg_{cfg.MODEL.CLIP_MODEL}_{backbone_name}")
cfg.DATASET.NAME = cfg.DATASET.NAME+f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else cfg.DATASET.NAME
seg_path = os.path.join(cfg.output_dir, cfg.DATASET.NAME, "seg_results", 
                        f"seed{cfg.seed}", results_name + f"_Prompt-{cfg.prompt_design}")

save_path = os.path.join(cfg.output_dir, cfg.DATASET.NAME, f"seg_results", 
                        f"seed{cfg.seed}", "test_" + f"Prompt-{cfg.prompt_design}.csv")

# Get list of GT and SEG files (allow multiple extensions)
valid_exts = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}

def strip_ext(fname):
    return os.path.splitext(fname)[0]  # name without extension

gt_files = {strip_ext(f): f for f in os.listdir(gt_path) if os.path.splitext(f)[1].lower() in valid_exts}
seg_files = {strip_ext(f): f for f in os.listdir(seg_path) if os.path.splitext(f)[1].lower() in valid_exts}

# Take intersection of names present in both
common_names = sorted(set(gt_files.keys()) & set(seg_files.keys()))
print(f"Found {len(common_names)} matching files")

# Build list of matching pairs
filenames = [(gt_files[name], seg_files[name]) for name in common_names]

# Initialize metrics dictionary
seg_metrics = OrderedDict(
    Name = list(),
    DSC = list(),
    NSD = list(),
)

# Compute metrics for each file
with tqdm(filenames) as pbar:
    for idx, (gt_name, seg_name) in enumerate(pbar):
        seg_metrics['Name'].append(gt_name)  # store GT name, or seg_name if you prefer
        gt_mask = cv2.imread(join(gt_path, gt_name), cv2.IMREAD_GRAYSCALE)
        seg_mask = cv2.imread(join(seg_path, seg_name), cv2.IMREAD_GRAYSCALE)
        seg_mask = cv2.resize(seg_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        # print(np.unique(gt_mask), np.unique(seg_mask))
        gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)[1]
        seg_mask = cv2.threshold(seg_mask, 127, 255, cv2.THRESH_BINARY)[1]
        gt_data = np.uint8(gt_mask)
        seg_data = np.uint8(seg_mask)

        # Check if both masks are completely empty (only background)
        if np.max(gt_data) == 0 and np.max(seg_data) == 0:
            DSC = 1.0
            NSD = 1.0
        else:
            gt_labels = np.unique(gt_data)
            seg_labels = np.unique(seg_data)
            # Skip background label (0)
            gt_labels = gt_labels[gt_labels != 0]
            seg_labels = seg_labels[seg_labels != 0]
            labels = np.union1d(gt_labels, seg_labels)

            DSC_arr = []
            NSD_arr = []

            for i in labels:
                if np.sum(gt_data == i) == 0 and np.sum(seg_data == i) == 0:
                    DSC_i = 1
                    NSD_i = 1
                elif np.sum(gt_data == i) == 0 and np.sum(seg_data == i) > 0:
                    DSC_i = 0
                    NSD_i = 0
                else:
                    i_gt, i_seg = gt_data == i, seg_data == i
                    DSC_i = compute_dice_coefficient(i_gt, i_seg)
                    surface_distances = compute_surface_distances(i_gt[..., None], i_seg[..., None], [1, 1, 1])
                    NSD_i = compute_surface_dice_at_tolerance(surface_distances, 2)

                DSC_arr.append(DSC_i)
                NSD_arr.append(NSD_i)

            DSC = np.mean(DSC_arr) if DSC_arr else 0.0
            NSD = np.mean(NSD_arr) if NSD_arr else 0.0

        seg_metrics['DSC'].append(round(DSC, 4))
        seg_metrics['NSD'].append(round(NSD, 4))

        pbar.set_postfix({
            'Mean DSC': f"{np.mean(seg_metrics['DSC']):.4f}",
            'Mean NSD': f"{np.mean(seg_metrics['NSD']):.4f}"
        })


# Save metrics to CSV
dataframe = pd.DataFrame(seg_metrics)
dataframe.to_csv(save_path, index=False)

# Calculate average
case_avg_DSC = dataframe['DSC'].mean()
case_avg_NSD = dataframe['NSD'].mean()

print(20 * '>')
print(f'Average DSC for {basename(seg_path)} {cfg.DATASET.NAME}: {case_avg_DSC * 100:.2f}%')
print(f'Average NSD for {basename(seg_path)} {cfg.DATASET.NAME}: {case_avg_NSD * 100:.2f}%')
print(20 * '<')