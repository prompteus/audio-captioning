CUDA_VISIBLE_DEVICES=7 python \
    audiocap/train_whisper_supervised.py \
    --checkpoint-dir-root="./checkpoints" \
    --clotho-dir="./data/clotho_v2.1/audiofolder" \
    --audioset-dir="./data/audioset_small/audiofolder" \
    --audiocaps-dir="./data/audiocaps/audiofolder" \
    --training-config="./configs/train_whisper_supervised_train_config.yaml"
