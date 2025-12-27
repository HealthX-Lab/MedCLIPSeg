# How to install datasets

Our study includes 16 biomedical image segmentation datasets. Place all the datasets in one directory under `data` to ease management. The file structure looks like

```
data/
├── <DATASET_NAME>/
│   ├── Prompts_Folder/
│   │   └── <prompt_files>        # text prompts (*.xlsx)
│   │
│   ├── Train_Folder/
│   │   ├── img/
│   │   │   └── <image_files>
│   │   └── label/
│   │       └── <mask_files>
│   │
│   ├── Val_Folder/
│   │   ├── img/
│   │   │   └── <image_files>
│   │   └── label/
│   │       └── <mask_files>
│   │
│   └── Test_Folder/
│       ├── img/
│       │   └── <image_files>
│       └── label/
│           └── <mask_files>
│
└── <DATASET_NAME_2>/
    └── (same structure as above)
```

# Dataset Organization
Each dataset is **split into training, validation, and testing splits**.

The `Prompts_Folder` contains the **text prompt files** associated with each dataset. These include:
- Prompt definitions used for **data-efficiency experiments** (e.g., 10%, 25%, 50% training and validation subsets)
- Additional **variant prompt designs** explored in the study, such as alternative phrasing and semantic formulations

These prompt files enable flexible evaluation under different supervision regimes.

## Dataset Summary

| **Dataset** | **Train** | **Validation** | **Test** | **Modality** | **Organ** |
|:--|:--:|:--:|:--:|:--|:--|
| [BUSI](https://pubmed.ncbi.nlm.nih.gov/31867417/) | (62, 156, 312) | (7, 19, 39) | 78 | Ultrasound | Breast |
| [BTMRI](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) | (273, 684, 1,369) | (132, 330, 660) | 1,005 | MRI | Brain |
| [ISIC](https://arxiv.org/abs/1902.03368) | (80, 202, 404) | (9, 22, 45) | 379 | Dermatoscopy | Skin |
| [Kvasir-SEG](https://arxiv.org/abs/1911.07069) | (80, 200, 400) | (10, 25, 50) | 100 | Endoscopy | Colon |
| [QaTa-COV19](https://arxiv.org/abs/2202.10185) | (571, 1,429, 2,858) | (142, 357, 714) | 2,113 | X-ray | Chest |
| [EUS](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/11583/2581321/Endoscopic-ultrasound-database-of-the-pancreas/10.1117/12.2581321.full) | (2,631, 6,579, 13,159) | (175, 439, 879) | 10,090 | Ultrasound | Pancreas |
| [BUSUC](https://www.sciencedirect.com/science/article/pii/S0952197623014768?via%3Dihub) | 567 | 122 | 122 | Ultrasound | Breast |
| [BUSBRA](https://aapm.onlinelibrary.wiley.com/doi/10.1002/mp.16812) | 1,311 | 282 | 282 | Ultrasound | Breast |
| [BUID](https://www.sciencedirect.com/science/article/pii/S0010482522011465) | 162 | 35 | 35 | Ultrasound | Breast |
| [UDIAT](https://www.sciencedirect.com/science/article/pii/S174680942030183X) | 113 | 25 | 25 | Ultrasound | Breast |
| [BRISC](https://arxiv.org/abs/2506.14318) | 4,000 | 1,000 | 1,000 | MRI | Brain |
| [UWaterlooSkinCancer](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6937161) | 132 | 0 | 41 | Dermatoscopy | Skin |
| [CVC-ColonDB](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7294676&tag=1) | 20 | 0 | 360 | Endoscopy | Colon |
| [CVC-ClinicDB](https://refbase.cvc.uab.es/files/BSF2015.pdf) | 490 | 61 | 61 | Endoscopy | Colon |
| [CVC-300](https://arxiv.org/abs/1612.00799) | 6 | 0 | 60 | Endoscopy | Colon |
| [BKAI](https://arxiv.org/abs/2107.05023) | 799 | 100 | 100 | Endoscopy | Colon |


### Download the datasets
All the datasets can be found on Hugging Face [here](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg). Download each dataset seperately:

- <b>BUSI</b> [Hugging Face](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/BUSI.zip)
- <b>BTMRI</b> [Hugging Face](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/BTMRI.zip)
- <b>ISIC</b> [Hugging Face](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/ISIC.zip)
- <b>Kvasir-SEG</b> [Hugging Face](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/Kvasir.zip)
- <b>QaTa-COV19</b> [Hugging Face](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/Covid19.zip)
- <b>EUS</b> [Drive](https://drive.google.com/drive/folders/10VFNadH6cD6bG3fD2Yf90iDv3JIqsIlv)
- <b>BUSUC</b> [Hugging Face](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/BUSUC.zip)
- <b>BUSBRA</b> [Hugging Face](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/BUSBRA.zip)
- <b>BUID</b> [Hugging Face](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/BUID.zip)
- <b>UDIAT</b> [Hugging Face](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/UDIAT.zip)
- <b>BRISC</b> [Hugging Face](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/BRISC.zip)
- <b>UWaterlooSkinCancer</b> [Hugging Face](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/UWaterlooSkinCancer.zip)
- <b>CVC-ColonDB</b> [Hugging Face](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/ColonDB.zip)
- <b>CVC-ClinicDB</b> [Hugging Face](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/ClinicDB.zip)
- <b>CVC-300</b> [Hugging Face](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/CVC300.zip)
- <b>BKAI</b> [Hugging Face](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg/blob/main/BKAI.zip)

After downloading each dataset, unzip and place each under `data` like the following

```
data/
├── BTMRI/
│   ├── Prompts_Folder/
│   │   └── <prompt_files>        # text prompts (*.xlsx)
│   │
│   ├── Train_Folder/
│   │   ├── img/
│   │   │   └── <image_files>
│   │   └── label/
│   │       └── <mask_files>
│   │
│   ├── Val_Folder/
│   │   ├── img/
│   │   │   └── <image_files>
│   │   └── label/
│   │       └── <mask_files>
│   │
│   └── Test_Folder/
│       ├── img/
│       │   └── <image_files>
│       └── label/
│           └── <mask_files>
```