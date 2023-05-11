
ADDITIONAL_INFO="_backup_dataset" # "_!!!TEST_first10!!!"

for CONFIG_FILE_NAME in predict_audiocaps_greedy_config 
do

    echo "Running inference with config file: ${CONFIG_FILE_NAME}.yaml"
    DATASET=audiocaps # clotho_v2.1
    RUN_NAME=ethereal-sponge-78_backup # magic-mountain-74 # rose-lake-75 # atomic-sky-43 # magic-mountain-74 # efficient-jazz-72 # dainty-yogurt-57 # mythical-trooper-62 (lora?)
    CHECKPOINT=2100 # 5000 # 1000 # 13500 # 6200  # 1000 # 2600
    SPLIT=valid
    OUT_FILE_NAME=${RUN_NAME}_${SPLIT}_${CONFIG_FILE_NAME}${ADDITIONAL_INFO}

    CUDA_VISIBLE_DEVICES=3 python \
        predict.py \
        --checkpoint ../../maratmp/audio-captioning/checkpoints/$RUN_NAME/checkpoint-${CHECKPOINT}  \
        --data  ../../maratmp/audio-captioning/data/$DATASET/audiofolder/$SPLIT \
        --output-file ../inference_outputs/${OUT_FILE_NAME}.csv \
        --config-file ../configs/hyperparameters_search/${CONFIG_FILE_NAME}.yaml
        # --take-first-n 10 # optional, for debugging purposes

    python compute_metrics.py \
        --predictions-path ../inference_outputs/${OUT_FILE_NAME}.csv \
        --labels-path ../../maratmp/audio-captioning/data/$DATASET/audiofolder/${SPLIT}/metadata.jsonl # ../../maratmp/audio-captioning/data/$DATASET/clotho_captions_${SPLIT}.csv
done


# alfrid paths
# --checkpoint ../checkpoints/$RUN_NAME/checkpoint-2600 \
# --data ../data/clotho_v2.1/$SPLIT \

# --labels-path ../data/clotho_v2.1/clotho_captions_${SPLIT}.csv
# ../../maratmp/audio-captioning/checkpoints/$RUN_NAME/checkpoint-$CHECKPOINT