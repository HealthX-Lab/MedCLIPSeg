#!/bin/bash

OUTPUT=$1
CHOICE=$2
SEED=$3

# Predefined dataset lists
BUS=("BUSI" "BUSBRA" "BUSUC" "BUID" "UDIAT")
ENDO=("Kvasir" "ColonDB" "ClinicDB" "CVC300" "BKAI")
DERM=("ISIC" "UWaterlooSkinCancer")
BRAIN=("BTMRI" "BRISC")

# Select dataset group
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

# ------------------------------------
# SOURCE = first dataset
# ------------------------------------
SOURCE=${DATASETS[0]}

echo "===================================="
echo "SOURCE DATASET (fully supervised): $SOURCE"
echo "TARGET DATASETS: ${DATASETS[@]:1}"
echo "===================================="

# ------------------------------------
# Train once on SOURCE
# ------------------------------------
CONFIG=${SOURCE}
python train.py \
    --config-file configs/${CONFIG}.yaml \
    --output-dir ${OUTPUT} \
    --seed ${SEED}

# ------------------------------------
# (1) Evaluate SOURCE → SOURCE (in-domain)
# ------------------------------------
echo "Evaluating SOURCE=${SOURCE} → TARGET=${SOURCE} (in-domain)"

python test.py \
    --config-file configs/${SOURCE}.yaml \
    --output-dir ${OUTPUT} \
    --source_dataset ${SOURCE} \
    --seed ${SEED}

python utils/eval.py \
    --config-file configs/${SOURCE}.yaml \
    --output-dir ${OUTPUT} \
    --seed ${SEED}

# ------------------------------------
# (2) Evaluate SOURCE → TARGET (out-of-domain)
# ------------------------------------
for TARGET in "${DATASETS[@]:1}"; do
    echo "Evaluating SOURCE=${SOURCE} → TARGET=${TARGET}"

    python test.py \
        --config-file configs/${TARGET}.yaml \
        --output-dir ${OUTPUT} \
        --source_dataset ${SOURCE} \
        --seed ${SEED}

    python utils/eval.py \
        --config-file configs/${TARGET}.yaml \
        --output-dir ${OUTPUT} \
        --seed ${SEED}
done
