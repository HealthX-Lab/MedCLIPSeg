OUTPUT=outputs_medclipseg_reproduce
SEED=666

for DATASET in BUSI BTMRI ISIC Kvasir Covid19 EUS
do
for DATA_PERCENTAGE in 10 25 50 100
do

echo "Processing dataset: $DATASET with data_percentage=${DATA_PERCENTAGE}"

CONFIG=${DATASET}

python train.py --config-file configs/${CONFIG}.yaml \
    --output-dir ${OUTPUT} \
    --data_percentage ${DATA_PERCENTAGE} \
    --seed ${SEED}

python test.py --config-file configs/${CONFIG}.yaml \
    --output-dir ${OUTPUT} \
    --source_dataset ${DATASET} \
    --data_percentage ${DATA_PERCENTAGE} \
    --seed ${SEED}

python utils/eval.py --config-file configs/${CONFIG}.yaml \
    --output-dir ${OUTPUT} \
    --data_percentage ${DATA_PERCENTAGE} \
    --seed ${SEED}

done
done

BUS=("BUSBRA" "BUSUC" "BUID" "UDIAT")
ENDO=("ColonDB" "ClinicDB" "CVC300" "BKAI")
DERM=("UWaterlooSkinCancer")
BRAIN=("BRISC")

for CHOICE in BUS ENDO DERM BRAIN
do

# Pick datasets & set SOURCE
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

for DATASET in "${DATASETS[@]}"; do
    echo "Processing SOURCE=${SOURCE} → TARGET=${DATASET}"
    CONFIG=${DATASET}
    python test.py --config-file configs/${CONFIG}.yaml \
        --output-dir ${OUTPUT} \
        --source_dataset ${SOURCE} \
        --seed ${SEED}

    python utils/eval.py --config-file configs/${CONFIG}.yaml \
        --output-dir ${OUTPUT} \
        --seed ${SEED}
done
done