
ADDITIONAL_INFO="_backup_dataset" # "_!!!TEST_first10!!!"

for CONFIG_FILE_NAME in  \
        predict_clotho_beam_classic5_config \
        predict_clotho_beam_classic10_config
do

    echo "Running inference with config file: ${CONFIG_FILE_NAME}.yaml"
    DATASET=clotho_v2.1_backup # clotho_v2.1
    RUN_NAME=fancy-galaxy-115 # peach-bush-111 # stilted-vortex-116 # trim-snow-114 # rose-lake-75 # atomic-sky-43 # magic-mountain-74 # efficient-jazz-72 # dainty-yogurt-57 # mythical-trooper-62 (lora?)
    CHECKPOINT=3900 # 10500 # 1500 # 2200 # 2000 # 1000 # 13500 # 6200  # 1000 # 2600
    SPLIT=validation
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
        --labels-path ../../maratmp/audio-captioning/data/$DATASET/clotho_captions_${SPLIT}.csv # ../../maratmp/audio-captioning/data/$DATASET/audiofolder/${SPLIT}/metadata.jsonl # # 
done


# alfrid paths
# --checkpoint ../checkpoints/$RUN_NAME/checkpoint-2600 \
# --data ../data/clotho_v2.1/$SPLIT \

# --labels-path ../data/clotho_v2.1/clotho_captions_${SPLIT}.csv
# ../../maratmp/audio-captioning/checkpoints/$RUN_NAME/checkpoint-$CHECKPOINT

# predict_clotho_beam_classic5_config