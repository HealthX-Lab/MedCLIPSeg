# Installation

This codebase is tested on Ubuntu 20.04.2 LTS with python 3.10. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -n medclipseg python=3.10 -y

# Activate the environment
conda activate medclipseg

# Install torch (requires version >= 2.4.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
```

* Clone MedCLIPSeg code repository and install requirements
```bash
# Clone MedCLIPSeg code base
git clone https://github.com/HealthX-Lab/MedCLIPSeg

cd MedCLIPSeg/
# Install requirements

pip install -r requirements.txt
```