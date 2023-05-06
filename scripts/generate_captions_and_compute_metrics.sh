RUN_NAME=vibrant-haze-32
CHECKPOINT_NUM=2400
CONFIG_YAML=generate_clotho_test
SPLIT=test # if changed, change the labels path too 
OUT_DIR=../inference_outputs

# CUDA_VISIBLE_DEVICES=1 
python evaluate_generate_captions.py \
    --load-checkpoint ../checkpoints/$RUN_NAME/checkpoint-$CHECKPOINT_NUM \
    --dataset-dir ../data/clotho_v2.1/audiofolder/ \
    --output-dir $OUT_DIR \
    --generate-config ../configs/$CONFIG_YAML.yaml \
    --split-type $SPLIT \
    --log-file $OUT_DIR/logs/runtimes.txt

PREDS_FILE=$OUT_DIR/${RUN_NAME}_clotho_${SPLIT}_${CONFIG_YAML}.jsonl

echo "Done generating captions and computing metrics, preds ID: $PREDS_FILE"

python evaluate_compute_metrics.py --predictions-path $PREDS_FILE \
                                   --labels-path ../data/clotho_v2.1/clotho_captions_evaluation.csv \
                                   --save-stats-path $OUT_DIR/logs/metrics_stats.txt