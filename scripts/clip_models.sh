#!/bin/bash

OUTPUT=output
SEED=666

# Predefined dataset lists
BUS=("BUSI" "BUSBRA" "BUSUC" "BUID" "UDIAT")
ENDO=("Kvasir" "ColonDB" "ClinicDB" "CVC300" "BKAI")
DERM=("ISIC" "UWaterlooSkinCancer")
BRAIN=("BTMRI" "BRISC")

for clip_model in unimedclip biomedclip clip pubmedclip
do
for CHOICE in BUS ENDO DERM BRAIN
do

# -------------------------------
# Select dataset group
# -------------------------------
if [ "$CHOICE" == "BUS" ]; then
    DATASETS=("${BUS[@]}")
elif [ "$CHOICE" == "ENDO" ]; then
    DATASETS=("${ENDO[@]}")
elif [ "$CHOICE" == "DERM" ]; then
    DATASETS=("${DERM[@]}")
elif [ "$CHOICE" == "BRAIN" ]; then
    DATASETS=("${BRAIN[@]}")
else
    echo "Unknown choice: $CHOICE"
    exit 1
fi

# -------------------------------
# SOURCE = first dataset
# -------------------------------
SOURCE=${DATASETS[0]}

echo "===================================="
echo "CLIP MODEL: ${clip_model}"
echo "SOURCE DATASET: ${SOURCE}"
echo "TARGET DATASETS: ${DATASETS[@]}"
echo "===================================="

# -------------------------------
# Conditional backbone override
# -------------------------------
OPTS="MODEL.CLIP_MODEL ${clip_model}"

if [ "$clip_model" == "pubmedclip" ]; then
    OPTS="${OPTS} MODEL.BACKBONE ViT-B/32"
fi

# -------------------------------
# Train once on SOURCE
# -------------------------------
python train.py \
    --config-file configs/${SOURCE}.yaml \
    --output-dir ${OUTPUT} \
    --seed ${SEED} \
    ${OPTS}

# -------------------------------
# (1) Evaluate SOURCE → SOURCE
# -------------------------------
echo "Evaluating SOURCE=${SOURCE} → TARGET=${SOURCE}"

python test.py \
    --config-file configs/${SOURCE}.yaml \
    --output-dir ${OUTPUT} \
    --source_dataset ${SOURCE} \
    --seed ${SEED} \
    ${OPTS}

python utils/eval.py \
    --config-file configs/${SOURCE}.yaml \
    --output-dir ${OUTPUT} \
    --seed ${SEED} \
    ${OPTS}

# -------------------------------
# (2) Evaluate SOURCE → TARGETS
# -------------------------------
for TARGET in "${DATASETS[@]:1}"; do
    echo "Evaluating SOURCE=${SOURCE} → TARGET=${TARGET}"

    python test.py \
        --config-file configs/${TARGET}.yaml \
        --output-dir ${OUTPUT} \
        --source_dataset ${SOURCE} \
        --seed ${SEED} \
        ${OPTS}

    python utils/eval.py \
        --config-file configs/${TARGET}.yaml \
        --output-dir ${OUTPUT} \
        --seed ${SEED} \
        ${OPTS}
done

done
done
