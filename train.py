import torch
import monai
from tqdm import tqdm
from statistics import mean
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.dataloader import DatasetSegmentation, RandomGenerator, ValGenerator
from trainers import *
import os
import argparse
import random
import numpy as np
from torch.nn.modules.loss import BCEWithLogitsLoss
import logging
from utils.main_utils import load_cfg_from_cfg_file, read_text

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--config-file",
    required=True,
    type=str,
    help="Path to config file",
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help="Whether to resume training"
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help="Random seed for reproducibility."
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


def print_args(args, cfg):
    logging.info("***************")
    logging.info("** Arguments **")
    logging.info("***************")
    logging.info("************")
    logging.info("** Config **")
    logging.info("************")
    logging.info(cfg)

def logger_config(log_path):
    loggerr = logging.getLogger()
    loggerr.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    loggerr.addHandler(handler)
    loggerr.addHandler(console)
    return loggerr

def calc_loss(low_res_logits, low_res_label_batch, ce_loss, dice_loss, cfg):
    loss_ce = ce_loss(low_res_logits, low_res_label_batch.float())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch)
    loss = cfg.TRAIN.DICE_WEIGHT * loss_dice + cfg.TRAIN.CE_WEIGHT * loss_ce
    return loss

# Validation function
def evaluate_validation_loss(model, val_dataloader, device):
    model.eval()
    val_losses = []
    dice_scores = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation"):
            images = batch["image"].to(device)
            masks = batch["ground_truth_mask"].to(device)
            text = batch["text_prompt"]

            logits = model(images, text=text, num_samples=1)[0]
            loss = calc_loss(logits, masks, ce_loss, dice_loss, cfg)
            val_losses.append(loss.item())

            # Compute Dice score manually
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Add channel dimension if missing
            if preds.ndim == 3:
                preds = preds.unsqueeze(1)
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)

            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            dice_scores.extend(dice.cpu().numpy())

    avg_loss = mean(val_losses)
    avg_dice = mean(dice_scores)
    model.train()
    return avg_loss, avg_dice

cfg = get_arguments()
cfg.DATASET.NAME = cfg.DATASET.NAME+f"_{cfg.data_percentage}" if cfg.data_percentage != 100 else cfg.DATASET.NAME
os.makedirs(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}"),exist_ok = True)

logger = logger_config(os.path.join(cfg.output_dir, cfg.DATASET.NAME, "trained_models", f"seed{cfg.seed}", "log.txt"))
logger.info("************")
logger.info("** Config **")
logger.info("************")
logger.info(cfg)
if cfg.seed >= 0:
    logger.info("Setting fixed seed: {}".format(cfg.seed))
    set_random_seed(cfg.seed)

# def worker_init_fn(worker_id):
#     random.seed(cfg.seed + worker_id)

def worker_init_fn(worker_id):
    seed = cfg.seed + worker_id
    random.seed(seed)
    np.random.seed(seed)

ce_loss = BCEWithLogitsLoss()
dice_loss = monai.losses.DiceLoss(
    include_background=False, 
    sigmoid=True, 
    reduction="mean"
)

train_tf = transforms.Compose([RandomGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])])
val_tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])

train_text_file = f"Train_text_{cfg.data_percentage}.xlsx" if cfg.data_percentage != 100 else "Train_text.xlsx"
val_text_file = f"Val_text_{cfg.data_percentage}.xlsx" if cfg.data_percentage != 100 else "Val_text.xlsx"
train_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + train_text_file)
val_text = read_text(cfg.DATASET.TEXT_PROMPT_PATH + val_text_file)

train_dataset = DatasetSegmentation(cfg.DATASET.TRAIN_PATH, cfg.DATASET.NAME, train_text, train_tf,
                                image_size=cfg.DATASET.SIZE)
val_dataset = DatasetSegmentation(cfg.DATASET.VAL_PATH, cfg.DATASET.NAME, val_text, val_tf, image_size=cfg.DATASET.SIZE)

train_dataloader = DataLoader(train_dataset,
                            batch_size=cfg.TRAIN.BATCH_SIZE,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                            num_workers=8,
                            pin_memory=True,)

val_dataloader = DataLoader(val_dataset,
                        batch_size=cfg.TRAIN.BATCH_SIZE,
                        shuffle=False,
                        worker_init_fn=worker_init_fn,
                        num_workers=8,
                        pin_memory=True)

if(cfg.MODEL.CLIP_MODEL == "unimedclip"):
    model = build_medclipseg_unimedclip(cfg)
elif(cfg.MODEL.CLIP_MODEL == "biomedclip"):
    model = build_medclipseg_biomedclip(cfg)
elif(cfg.MODEL.CLIP_MODEL == "clip"):
    model = build_medclipseg_clip(cfg)
elif(cfg.MODEL.CLIP_MODEL == "pubmedclip"):
    model = build_medclipseg_pubmedclip(cfg)

enabled = set()
for name, param in model.named_parameters():
    if param.requires_grad:
        enabled.add(name)

logger.info(f"Parameters to be updated: {enabled}")
logger.info(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Initialize optimizer and Loss
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LEARNING_RATE)
num_epochs = cfg.TRAIN.NUM_EPOCHS

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,     # decay over all epochs
    eta_min=1e-4
)

backbone_name = cfg.MODEL.BACKBONE.replace("/", "-")

results_name = (
    f"MedCLIPSeg_"
    f"{cfg.MODEL.CLIP_MODEL}_"
    f"{backbone_name}"
)

# Resume functionality
resume_path = os.path.join(
            cfg.output_dir,
            cfg.DATASET.NAME,
            "trained_models",
            f"{results_name}_latest.pth")

start_epoch = 0
best_loss = float("inf")
best_dice = 0

if cfg.resume and os.path.exists(resume_path):
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint.get("scheduler", {}))
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["best_loss"]
    logger.info(f"Loaded checkpoint from epoch {start_epoch}, best loss: {best_loss:.4f}")

# Set model to train and into the device
model.train()
model.to(cfg.MODEL.DEVICE)

total_loss = []

for epoch in range(start_epoch, num_epochs):
    epoch_losses = []

    for i, batch in enumerate(tqdm(train_dataloader)):

        seg_logits, clip_loss = model(
            image=batch["image"].to(cfg.MODEL.DEVICE),
            text=batch["text_prompt"]
        )
        total_loss = 0
        loss = calc_loss(seg_logits, batch['ground_truth_mask'].to(cfg.MODEL.DEVICE), ce_loss, dice_loss, cfg)
        loss += cfg.TRAIN.CLIP_WEIGHT*clip_loss
        optimizer.zero_grad()
        loss.backward()
        # Optimize
        optimizer.step()
        epoch_losses.append(loss.item())

    # Scheduler step at the end of the epoch
    scheduler.step()

    # End of epoch operations
    mean_epoch_loss = mean(epoch_losses)
    # Validation phase
    mean_val_loss, mean_val_dice = evaluate_validation_loss(model, val_dataloader, cfg.MODEL.DEVICE)
    logger.info(f'EPOCH: {epoch+1} | Training Loss: {mean_epoch_loss:.4f} | Validation Loss: {mean_val_loss:.4f}')

    # Save the best model based on validation loss
    if mean_val_dice > best_dice:
        logger.info(f"New best Dice: {best_dice:.4f} → {mean_val_dice:.4f}")
        best_dice = mean_val_dice
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_dice": best_dice,
        }, os.path.join(
            cfg.output_dir,
            cfg.DATASET.NAME,
            "trained_models",
            f"seed{cfg.seed}",
            f"{results_name}_best_dice.pth"
        ))
    else:
        logger.info(f"Dice: {mean_val_dice:.4f}")

    # Save the latest model
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_loss": best_loss
    }, 
    os.path.join(
    cfg.output_dir,
    cfg.DATASET.NAME,
    "trained_models",
    f"seed{cfg.seed}",
    f"{results_name}_latest.pth")
    )