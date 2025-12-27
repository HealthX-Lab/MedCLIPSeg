import os
import argparse
import shutil
from huggingface_hub import hf_hub_download

REPO_ID = "TahaKoleilat/MedCLIPSeg"
CKPT_NAME = "MedCLIPSeg_unimedclip_ViT-B-16_latest.pth"

def main(args):
    dataset = args.dataset
    data_percentage = args.data_percentage
    seed = args.seed

    # ----------------------------------
    # Resolve dataset directory name
    # ----------------------------------
    if data_percentage is None or data_percentage == 100:
        dataset_dir = dataset
        hf_subdir = dataset
    else:
        dataset_dir = f"{dataset}_{data_percentage}"
        hf_subdir = f"{dataset}_{data_percentage}"

    # ----------------------------------
    # Target directory (local)
    # ----------------------------------
    target_dir = f"outputs_medclipseg/{dataset_dir}/trained_models/seed{seed}"
    os.makedirs(target_dir, exist_ok=True)

    target_ckpt = os.path.join(target_dir, CKPT_NAME)

    # ----------------------------------
    # HuggingFace path (remote)
    # ----------------------------------
    hf_path = f"{hf_subdir}/trained_models/seed{seed}/{CKPT_NAME}"

    # ----------------------------------
    # Skip if already exists
    # ----------------------------------
    if os.path.exists(target_ckpt):
        print(f"✅ Found checkpoint: {target_ckpt}")
        return

    print(
        f"⬇️  Downloading checkpoint for {dataset}"
        f"{'' if data_percentage in [None, 100] else f' ({data_percentage}%)'}"
    )

    # ----------------------------------
    # Download to HF cache
    # ----------------------------------
    cached_file = hf_hub_download(
        repo_id=REPO_ID,
        filename=hf_path,
        local_dir=None,  # use HF cache
        local_dir_use_symlinks=False,
    )

    # ----------------------------------
    # Copy to correct outputs directory
    # ----------------------------------
    shutil.copy(cached_file, target_ckpt)

    print(f"✅ Checkpoint placed at: {target_ckpt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_percentage", type=int, default=None)
    parser.add_argument("--seed", type=int, default=666)
    args = parser.parse_args()

    main(args)
