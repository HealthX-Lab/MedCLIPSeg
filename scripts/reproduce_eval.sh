#!/bin/bash

OUTPUT=outputs_medclipseg
SEED=666

############################################
# Data-efficiency & Fully-supervised
############################################
for DATASET in BUSI BTMRI ISIC Kvasir Covid19 EUS
do
for DATA_PERCENTAGE in 10 25 50 100
do

echo "=============================================="
echo "Processing dataset: ${DATASET} (${DATA_PERCENTAGE}%)"
echo "=============================================="

CONFIG=${DATASET}

# --------------------------------------------------
# Download checkpoint if missing
# --------------------------------------------------
python utils/download_ckpt.py \
    --dataset ${DATASET} \
    --data_percentage ${DATA_PERCENTAGE} \
    --seed ${SEED}

# --------------------------------------------------
# Run inference
# --------------------------------------------------
python test.py --config-file configs/${CONFIG}.yaml \
    --output-dir ${OUTPUT} \
    --source_dataset ${DATASET} \
    --data_percentage ${DATA_PERCENTAGE} \
    --seed ${SEED}

# --------------------------------------------------
# Run evaluation
# --------------------------------------------------
python utils/eval.py --config-file configs/${CONFIG}.yaml \
    --output-dir ${OUTPUT} \
    --data_percentage ${DATA_PERCENTAGE} \
    --seed ${SEED}

done
done


############################################
# Domain Generalization
############################################
BUS=("BUSBRA" "BUSUC" "BUID" "UDIAT")
ENDO=("ColonDB" "ClinicDB" "CVC300" "BKAI")
DERM=("UWaterlooSkinCancer")
BRAIN=("BRISC")

for CHOICE in BUS ENDO DERM BRAIN
do

# --------------------------------------------------
# Pick source and targets
# --------------------------------------------------
if [ "$CHOICE" == "BUS" ]; then
    DATASETS=("${BUS[@]}")
    SOURCE="BUSI"
elif [ "$CHOICE" == "ENDO" ]; then
    DATASETS=("${ENDO[@]}")
    SOURCE="Kvasir"
elif [ "$CHOICE" == "DERM" ]; then
    DATASETS=("${DERM[@]}")
    SOURCE="ISIC"
elif [ "$CHOICE" == "BRAIN" ]; then
    DATASETS=("${BRAIN[@]}")
    SOURCE="BTMRI"
fi

# --------------------------------------------------
# Download SOURCE checkpoint once
# --------------------------------------------------
python utils/download_ckpt.py \
    --dataset ${SOURCE} \
    --seed ${SEED}

for DATASET in "${DATASETS[@]}"; do

    echo "=============================================="
    echo "Processing SOURCE=${SOURCE} → TARGET=${DATASET}"
    echo "=============================================="

    CONFIG=${DATASET}

    # --------------------------------------------------
    # Run inference
    # --------------------------------------------------
    python test.py --config-file configs/${CONFIG}.yaml \
        --output-dir ${OUTPUT} \
        --source_dataset ${SOURCE} \
        --seed ${SEED}

    # --------------------------------------------------
    # Run evaluation
    # --------------------------------------------------
    python utils/eval.py --config-file configs/${CONFIG}.yaml \
        --output-dir ${OUTPUT} \
        --seed ${SEED}

done
done
