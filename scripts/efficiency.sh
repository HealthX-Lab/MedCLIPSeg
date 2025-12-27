OUTPUT=$1
DATASET=$2
DATA_PERCENTAGE=$3
SEED=$4

CONFIG=${DATASET}

echo "Processing dataset: $DATASET using config: ${CONFIG}.yaml with data_percentage=${DATA_PERCENTAGE}"

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