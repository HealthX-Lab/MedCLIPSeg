<div align="center">
  
# MedCLIPSeg: Probabilistic Vision–Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation

<h3>CVPR 2026</h3>

**[Health-X Lab](http://www.healthx-lab.ca/)** | **[IMPACT Lab](https://users.encs.concordia.ca/~impact/)** 

[Taha Koleilat](https://tahakoleilat.github.io/), 
[Hojat Asgariandehkordi](https://scholar.google.com/citations?user=ndXNye4AAAAJ&hl=en), 
[Omid Nejati Manzari](https://omid-nejati.github.io/), 
[Berardino Barile](https://scholar.google.com/citations?user=odmpMGcAAAAJ&hl=en), 
[Yiming Xiao](https://yimingxiao.weebly.com/curriculum-vitae.html)<sup>†</sup>, 
[Hassan Rivaz](https://users.encs.concordia.ca/~hrivaz/)<sup>†</sup>

[![paper](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2602.20423)
[![Website](https://img.shields.io/badge/Website-Project%20Page-brightgreen.svg)](https://tahakoleilat.github.io/MedCLIPSeg)
[![Datasets](https://img.shields.io/badge/Datasets-Access-yellow.svg)](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg)
[![Models](https://img.shields.io/badge/Models-Reproduce-orange.svg)](https://huggingface.co/TahaKoleilat/MedCLIPSeg)
[![BibTeX](https://img.shields.io/badge/BibTeX-Cite-blueviolet.svg)](#citation)

† *Co-senior authors*
</div>

## Overview

![main figure](assets/overview.png)
> **<p align="justify"> Abstract:** *Medical image segmentation remains challenging due to limited annotations for training, ambiguous anatomical features, and domain shifts. While vision–language models such as **CLIP** offer strong cross-modal representations, their potential for dense, text-guided medical image segmentation remains underexplored. We present **MedCLIPSeg**, a novel framework that adapts CLIP for **robust, data-efficient, and uncertainty-aware** medical image segmentation. Our approach leverages **patch-level CLIP embeddings** through **probabilistic cross-modal attention**, enabling bidirectional interaction between image and text tokens and explicit modeling of predictive uncertainty. Together with a **soft patch-level contrastive loss** that encourages nuanced semantic learning across diverse textual prompts, **MedCLIPSeg** improves data efficiency and domain generalizability. Extensive experiments across **16 datasets**, spanning **five imaging modalities** and **six organs**, demonstrate that **MedCLIPSeg** outperforms prior methods in **accuracy, efficiency, and robustness**, while providing **interpretable uncertainty maps** that highlight the local reliability of segmentation results. This work demonstrates the potential of **probabilistic vision–language modeling** for text-driven medical image segmentation.* </p>

## Method

<p align="center">
  <img src="assets/MedCLIPSeg.png" width="100%" />
  <br>
  <em>Overall architecture of MedCLIPSeg. The framework integrates probabilistic vision–language fusion into a CLIP-based segmentation pipeline.</em>
</p>

<br>

<p align="center">
  <img src="assets/MedCLIPSeg_PVL.png" width="85%" />
  <br>
  <em>Schematic illustration of the proposed Probabilistic Vision–Language (PVL) adapters used for bidirectional cross-modal interaction.</em>
</p>


1) **Bidirectional Vision–Language Fusion**: Introduce representation-level fusion modules that enable efficient bidirectional interaction between image and text features while keeping CLIP encoders frozen, improving data efficiency and robustness.

2) **Probabilistic Cross-Modal Attention**: Model vision–language attention using variational Key–Value formulations to capture uncertainty, leading to improved segmentation accuracy and cross-domain generalization.

3) **Pixel-Level Uncertainty Estimation**: Generate dense uncertainty maps by sampling attention Values from learned probability distributions, providing intuitive reliability estimates for clinical interpretation.

4) **Extensive Multi-Modal Segmentation Evaluation**: Conduct comprehensive evaluation against state-of-the-art methods across 5 imaging modalities and 6 organs and 16 datasets, assessing data efficiency, domain generalization, and the contribution of individual model components.

## Results
Results reported below show DSC scores (%) for data efficiency and domain generalization evaluation benchmarks across 16 biomedical image segmentation datasets averaged.

### Data-Efficiency Evaluation

| **Method** | **10% Data** | **25% Data** | **50% Data** | **100% Data** |
|-----------|:------------:|:------------:|:------------:|:-------------:|
| [UNet](https://arxiv.org/abs/1505.04597) | 60.95 | 62.74 | 71.61 | 78.49 |
| [UNet++](https://arxiv.org/abs/1807.10165) | 63.72 | 65.86 | 73.15 | 78.44 |
| [DeepLabv3](https://arxiv.org/abs/1706.05587) | 61.32 | 65.39 | 68.58 | 73.28 |
| [Attention U-Net](https://arxiv.org/abs/1804.03999) | 62.78 | 64.97 | 71.34 | 76.30 |
| [nnU-Net](https://arxiv.org/abs/1809.10486) | 73.45 | 76.73 | 78.86 | 81.40 |
| [Swin-UNet](https://arxiv.org/abs/2105.05537) | 53.04 | 54.69 | 55.89 | 65.03 |
| [TransUNet](https://arxiv.org/abs/2102.04306) | 52.69 | 55.25 | 55.22 | 67.22 |
| [LViT](https://arxiv.org/abs/2206.14718) | 66.51 | 75.66 | 78.88 | 83.35 |
| [Ariadne’s Thread](https://arxiv.org/abs/2307.03942) | 61.34 | 63.09 | 65.65 | 70.07 |
| [EoMT-CLIP](https://arxiv.org/abs/2503.19108) | 74.07 | 76.29 | 79.19 | 82.93 |
| [CLIPSeg](https://arxiv.org/abs/2112.10003) | 74.66 | 78.31 | 79.63 | 84.87 |
| [DenseCLIP](https://arxiv.org/abs/2112.01518) | 67.84 | 70.23 | 72.09 | 74.19 |
| [ZegCLIP](https://arxiv.org/abs/2212.03588) | 61.25 | 72.46 | 76.21 | 78.98 |
| [SAN](https://arxiv.org/abs/2302.12242) | 74.13 | 76.13 | 78.80 | 81.62 |
| [MaPLe](https://arxiv.org/abs/2210.03117) | 66.27 | 71.53 | 74.60 | 74.60 |
| [MaPLe + Decoder](https://arxiv.org/abs/2210.03117) | 74.81 | 79.64 | 82.81 | 84.94 |
| [VLSM-Adapter](https://arxiv.org/abs/2405.06196) | 74.47 | 77.63 | 80.83 | 83.85 |
| [CausalCLIPSeg](https://arxiv.org/abs/2503.15949) | 71.19 | 75.42 | 78.60 | 81.34 |
| [CAT-Seg](https://arxiv.org/abs/2303.11797) | *78.76* | *81.12* | *83.32* | *85.90* |
| **[MedCLIPSeg (Ours)](https://arxiv.org/abs/XXXX.XXXXX)** | **81.10** | **85.08** | **87.18** | **88.66** |

### Domain Generalization

| **Method** | **ID** | **OOD** | **HM** |
|-----------|:--------------------:|:---------------------:|:-----------------:|
| [LViT](https://arxiv.org/abs/2206.14718) | 83.31 | 64.99 | 73.02 |
| [Ariadne’s Thread](https://arxiv.org/abs/2307.03942) | 68.25 | 27.23 | 38.93 |
| [CLIPSeg](https://arxiv.org/abs/2112.10003) | 84.95 | 69.22 | 76.28 |
| [DenseCLIP](https://arxiv.org/abs/2112.01518) | 77.69 | 58.11 | 66.49 |
| [ZegCLIP](https://arxiv.org/abs/2212.03588) | 77.16 | 61.33 | 68.34 |
| [SAN](https://arxiv.org/abs/2302.12242) | 84.45 | 69.87 | 76.47 |
| [MaPLe](https://arxiv.org/abs/2210.03117) | 76.55 | 59.30 | 66.83 |
| [MaPLe + Decoder](https://arxiv.org/abs/2210.03117) | 84.78 | 66.85 | 74.76 |
| [VLSM-Adapter](https://arxiv.org/abs/2405.06196) | 85.78 | 73.28 | 79.04 |
| [CausalCLIPSeg](https://arxiv.org/abs/2503.15949) | 81.52 | 53.86 | 64.86 |
| [CAT-Seg](https://arxiv.org/abs/2303.11797) | 86.10 | 74.57 | 79.92 |
| **[MedCLIPSeg (Ours)](https://arxiv.org/abs/XXXX.XXXXX)** | **89.11** | **79.02** | **83.76** |

### Segmentation and Uncertainty Visualization

<p align="center">
  <img src="assets/MedCLIPSeg_Seg.png" width="100%" />
</p>

<p align="center">
  <em>
    Uncertainty peaks along lesion boundaries and remains consistent across diverse datasets, indicating reliable calibration and generalization.
    In-distribution (ID) data are shown in <b style="color:#0000FF;">blue</b>, while out-of-distribution (OOD) data are shown in <b style="color:#FF0000;">red</b>.
  </em>
</p>

## Model Checkpoints
All the checkpoints can be found on the official [Hugging Face repo](https://huggingface.co/TahaKoleilat/MedCLIPSeg) for the Data Efficiency and Domain Generalization evaluation benchmarks. Take a look [here](https://github.com/HealthX-Lab/MedCLIPSeg/blob/main/assets/RUN.md#2-running-evaluation-from-given-checkpoints) to see how to run and reproduce all the results.

## Installation 
For installation and other package requirements, please follow the instructions detailed in [INSTALL.md](assets/INSTALL.md). 

## Data preparation
Please follow the instructions at [DATASETS.md](assets/DATASETS.md) to prepare all datasets.

## Training and Evaluation
Please refer to the [RUN.md](assets/RUN.md) for detailed instructions on training, evaluating and reproducing the results using our pre-trained models.

<hr />

## Citation
If you use our work, please consider citing:
```bibtex
@article{koleilat2026medclipseg,
  title={MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation},
  author={Koleilat, Taha and Asgariandehkordi, Hojat and Manzari, Omid Nejati and Barile, Berardino and Xiao, Yiming and Rivaz, Hassan},
  journal={arXiv preprint arXiv:2602.20423},
  year={2026}
}
```

## Acknowledgements
We are grateful to the authors of [CLIP](https://github.com/openai/CLIP), [MaPLe](https://github.com/muzairkhattak/multimodal-prompt-learning), and [LViT](https://github.com/HUANGLIZI/LViT) for making their code publicly available. If you use our model or code, we kindly request that you also consider citing these foundational works.
