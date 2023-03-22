CUDA_VISIBLE_DEVICES=6 python \
    audiocap/train_whisper_supervised.py \
    --architecture-name "openai/whisper-small" \
    --checkpoint-dir-root "./checkpoints" \
    --no-use-pretrained-whisper-encoder \
    --no-use-pretrained-whisper-decoder \
    --clotho-dir "./data/clotho_v2.1" \
    --log-preds-every-n-steps 500 \
    --limit-val-split-size 250 \
    --training-args-config "./configs/train_whisper_supervised_train_args.yaml"
