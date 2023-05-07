
ADDITIONAL_INFO="" # "_!!!TEST_first10!!!"

for CONFIG_FILE_NAME in predict_clotho_beam_multinomial10_topk3_config \

do
    echo "Running inference with config file: ${CONFIG_FILE_NAME}.yaml"
    RUN_NAME=dainty-yogurt-57
    SPLIT=validation
    OUT_FILE_NAME=${RUN_NAME}_${SPLIT}_${CONFIG_FILE_NAME}${ADDITIONAL_INFO}

    CUDA_VISIBLE_DEVICES=1 python \
        predict.py \
        --checkpoint ../../maratmp/audio-captioning/checkpoints/$RUN_NAME/checkpoint-2600 \
        --data  ../../maratmp/audio-captioning/data//clotho_v2.1/audiofolder/$SPLIT \
        --output-file ../inference_outputs/${OUT_FILE_NAME}.csv \
        --config-file ../configs/hyperparameters_search/${CONFIG_FILE_NAME}.yaml
        # --take-first-n 10 # optional, for debugging purposes

    python compute_metrics.py \
        --predictions-path ../inference_outputs/${OUT_FILE_NAME}.csv \
        --labels-path ../../maratmp/audio-captioning/data/clotho_v2.1/clotho_captions_${SPLIT}.csv
done


# alfrid paths
# --checkpoint ../checkpoints/$RUN_NAME/checkpoint-2600 \
# --data ../data/clotho_v2.1/$SPLIT \

# --labels-path ../data/clotho_v2.1/clotho_captions_${SPLIT}.csv