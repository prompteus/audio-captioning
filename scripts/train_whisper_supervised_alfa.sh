CUDA_VISIBLE_DEVICES=1 python \
    audiocap/train_whisper_supervised.py \
    --checkpoint-dir-root="./checkpoints" \
    --clotho-dir="../maratmp/audio-captioning/data/clotho_v2.1_backup/audiofolder" \
    --audioset-dir="../maratmp/audio-captioning/data/audioset_small/audiofolder" \
    --audiocaps-dir="../maratmp/audio-captioning/data/audiocaps/audiofolder" \
    --training-config="./configs/pretrain_1:12:3_large_half_scratch_config.yaml" \
    --wandb-group="pretraining"
