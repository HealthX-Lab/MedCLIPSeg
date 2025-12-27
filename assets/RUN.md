# Training and Evaluation

We provide bash scripts in [scripts/](../scripts) for each technique including data efficiency and domain generalization evl.
Make sure to setup the dataset paths in `data` and run the commands from the main directory `MedCLIPSeg/`.
Below we provide training and evaluation instructions for MedCLIPSeg. The same instructions applies for all other techniques.


### Training time and compute
We train MedCLIPSeg on each dataset with a batch size of 24 using a **single** NVIDIA A100 GPU (40 GB RAM).

### Configs

The default training settings are provided in the config files at [configs/](). All hyper-parameters can be modified using this config file.

## MedCLIPSeg

#### (1) Data Efficiency evaluation setting

Below, we provide instructions to train MedCLIPSeg on any dataset. 

```bash
# All possible dataset values include [BUSI, BTMRI, ISIC, Kvasir, Covid19, EUS]

# Data Percentage include [10, 25, 50, 100]

# trains and evaluates in data efficiency setting on a particular seed
bash scripts/efficiency.sh <output directory> <dataset> <data percentage> <seed>
# Example on BTMRI using 25% of the data using seed 666
bash scripts/efficiency.sh output BTMRI 25 666
```

The above steps can be repeated for other individual datasets.

#### (2) Domain generalization setting

```bash
# All possible domain choices include [BUS, ENDO, DERM, BRAIN]

# trains and evaluates on ID and OOD data
bash scripts/domain_generalization.sh <output directory> <domain choice> <seed>
# Example on Brain MRI domain using seed 666
bash scripts/domain_generalization.sh output BRAIN 666
```

#### Output Results: 
Once the above trainings and evaluations are completed, the `output/` directory should have the following structure:

```
output
|–– BTMRI_25/
|   |–– seg_results/
|   |   |–– seed666/
|   |   |   |–– MedCLIPSeg_unimedclip_ViT-B-16_Prompt-original/
|   |   |   |   |–– <segmentation results>
|   |–– trained_models/
|   |   |–– seed666/
|   |   |   |–– log.txt/
|   |   |   |–– MedCLIPSeg_unimedclip_ViT-B-16_best_dice.pth/
|   |   |   |–– MedCLIPSeg_unimedclip_ViT-B-16_latest.pth/
|   |–– unc_results/
|   |   |–– seed66/
|   |   |   |–– MedCLIPSeg_unimedclip_ViT-B-16_Prompt-original/
|   |   |   |   |–– <uncertainty results>
```

The above steps can be repeated for other individual datasets.

#### Reproducing Results

Our trained model checkpoints can be found on Hugging Face [here](https://huggingface.co/TahaKoleilat/MedCLIPSeg)

Run the following scripts to use the checkpoints and get testing results:

##### (1) Running on your environment

```bash
bash scripts/reproduce.sh
```

##### (2) Running evaluation from given checkpoints

Note: This script automatically checks for the required **MedCLIPSeg** pretrained checkpoints in the local `outputs_medclipseg/` directory.

If a checkpoint is missing, it is automatically downloaded from the official [Hugging Face repository](https://huggingface.co/TahaKoleilat/MedCLIPSeg) and placed in the correct directory structure before evaluation begins.

```bash
bash scripts/reproduce_eval.sh
```

##### (3) Text Prompt Ablation

```bash
bash scripts/text_prompts.sh
```

##### (4) CLIP Model Ablation

```bash
bash scripts/clip_model.sh
```