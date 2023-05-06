CUDA_VISIBLE_DEVICES=1 OUTFILE=python evaluate_generate_captions.py \
    --load-checkpoint ../checkpoints/stoic-totem-29/checkpoint-18900 \
    --dataset-dir ../data/clotho_v2.1/audiofolder/ \
    --output-dir ../inference_outputs \
    --generate-config ../configs/generate_clotho_test.yaml \
    --split-type test