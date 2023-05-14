# pretrain
CUDA_VISIBLE_DEVICES=1 python \
    audiocap/train_whisper_supervised.py \
    --load-checkpoint="checkpoints/usual-dew-70/checkpoint-23400/" \
    --checkpoint-dir-root="./checkpoints" \
    --clotho-dir="../maratmp/audio-captioning/data/clotho_v2.1_backup/audiofolder" \
    --audioset-dir="../maratmp/audio-captioning/data/audioset_small/audiofolder" \
    --audiocaps-dir="../maratmp/audio-captioning/data/audiocaps/audiofolder" \
    --training-config="./configs/pretrain_3on1_tiny_config.yaml" \
    --wandb-group="finetuning"

# 
CUDA_VISIBLE_DEVICES=1 nice -n 5 python \
    audiocap/train_whisper_supervised.py \
    --checkpoint-dir-root="../maratmp/audio-captioning/checkpoints" \
    --clotho-dir="../maratmp/audio-captioning/data/clotho_v2.1_backup/audiofolder" \
    --audioset-dir="../maratmp/audio-captioning/data/audioset_small/audiofolder" \
    --audiocaps-dir="../maratmp/audio-captioning/data/audiocaps/audiofolder" \
    --training-config="./configs/pretrain_1:12:3_tiny_config.yaml" \
    --wandb-group="pretraining"

CUDA_VISIBLE_DEVICES=2 nice -n 5 python \
    audiocap/train_whisper_supervised.py \
    --checkpoint-dir-root="../maratmp/audio-captioning/checkpoints" \
    --clotho-dir="../maratmp/audio-captioning/data/clotho_v2.1_backup/audiofolder" \
    --audioset-dir="../maratmp/audio-captioning/data/audioset_small/audiofolder" \
    --audiocaps-dir="../maratmp/audio-captioning/data/audiocaps/audiofolder" \
    --training-config="./configs/pretrain_1:12:3_small_config.yaml" \
    --wandb-group="pretraining"


# clotho clever freeze, new augment
CUDA_VISIBLE_DEVICES=1 nice -n 5 python \
    audiocap/train_whisper_supervised.py \
    --checkpoint-dir-root="../maratmp/audio-captioning/checkpoints" \
    --clotho-dir="../maratmp/audio-captioning/data/clotho_v2.1_backup/audiofolder" \
    --training-config="./configs/finetune_large_augment_slice_config.yaml" \
    --load-checkpoint="../maratmp/audio-captioning/checkpoints/stoic-totem-29/checkpoint-18900" \
    --wandb-group="finetuning"


# clever freeze no new augment
CUDA_VISIBLE_DEVICES=2 nice -n 5 python \
    audiocap/train_whisper_supervised.py \
    --checkpoint-dir-root="../maratmp/audio-captioning/checkpoints" \
    --clotho-dir="../maratmp/audio-captioning/data/clotho_v2.1_backup/audiofolder" \
    --training-config="./configs/finetune_large_augment_lora-q-v-fc1_config.yaml" \
    --load-checkpoint="../maratmp/audio-captioning/checkpoints/stoic-totem-29/checkpoint-18900" \
    --wandb-group="finetuning"







pgrep -w python -u xhajek9 | xargs -i{} taskset -cp 0-64 {}

    pretrain_3on1_tiny_config
    pretrain_1:12:3_large_half_scratch_config