
ADDITIONAL_INFO="_!!!TEST!!!"

for CONFIG_FILE_NAME in predict_clotho_beam_diverse_5_5_config \
                        predict_clotho_beam_multinomial5_config \
                        predict_clotho_contrastive_search_config \
                        predict_clotho_greedy_config \
                        predict_clotho_sample_topk_config
do
    RUN_NAME=dainty-yogurt-57
    SPLIT=validation
    OUT_FILE_NAME=${RUN_NAME}_${SPLIT}_${CONFIG_FILE_NAME}${ADDITIONAL_INFO}

    # CUDA_VISIBLE_DEVICES="..." 
    python \
    ../audiocap/predict.py \
    --checkpoint ../checkpoints/$RUN_NAME/checkpoint-2600 \
    --data ../data/clotho_v2.1/$SPLIT \
    --output-file ../inference_outputs/${OUT_FILE_NAME}.csv \
    --config-file ../configs/${CONFIG_FILE_NAME}.yaml \
    --log-file ../inference_outputs/${OUT_FILE_NAME}_log.csv \
    --take-first-n 10 # optional, for debugging purposes
done