#!/bin/bash
OUTPUT=output
SEED=666

# Predefined dataset lists
BUS=("BUSI" "BUSBRA" "BUSUC" "BUID" "UDIAT")
ENDO=("Kvasir" "ColonDB" "ClinicDB" "CVC300" "BKAI")
DERM=("ISIC" "UWaterlooSkinCancer")
BRAIN=("BTMRI" "BRISC")

for CHOICE in BRAIN BUS ENDO DERM
do

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
echo "TARGET DATASETS: ${DATASETS[@]}"
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
for prompt_design in original contradictory missing_location overdescriptive underdescriptive
do
echo "Evaluating SOURCE=${SOURCE} → TARGET=${SOURCE} (in-domain) with prompt_design=${prompt_design}"

python test.py \
    --config-file configs/${SOURCE}.yaml \
    --output-dir ${OUTPUT} \
    --source_dataset ${SOURCE} \
    --prompt_design ${prompt_design} \
    --seed ${SEED}

python utils/eval.py \
    --config-file configs/${SOURCE}.yaml \
    --output-dir ${OUTPUT} \
    --prompt_design ${prompt_design} \
    --seed ${SEED}
done
# ------------------------------------
# (2) Evaluate SOURCE → TARGET (out-of-domain)
# ------------------------------------
for prompt_design in original contradictory missing_location overdescriptive underdescriptive
do
for TARGET in "${DATASETS[@]:1}"; do
    echo "Evaluating SOURCE=${SOURCE} → TARGET=${TARGET} with prompt_design=${prompt_design}"

    python test.py \
        --config-file configs/${TARGET}.yaml \
        --output-dir ${OUTPUT} \
        --source_dataset ${SOURCE} \
        --prompt_design ${prompt_design} \
        --seed ${SEED}

    python utils/eval.py \
        --config-file configs/${TARGET}.yaml \
        --output-dir ${OUTPUT} \
        --prompt_design ${prompt_design} \
        --seed ${SEED}
done
done
done