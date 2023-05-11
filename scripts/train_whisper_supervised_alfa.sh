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
CUDA_VISIBLE_DEVICES=0 python \
    audiocap/train_whisper_supervised.py \
    --checkpoint-dir-root="./checkpoints" \
    --clotho-dir="../maratmp/audio-captioning/data/clotho_v2.1_backup/audiofolder" \
    --audioset-dir="../maratmp/audio-captioning/data/audioset_small/audiofolder" \
    --audiocaps-dir="../maratmp/audio-captioning/data/audiocaps/audiofolder" \
    --training-config="./configs/.yaml" \
    --wandb-group="pretraining"

# clotho backup finetuning
CUDA_VISIBLE_DEVICES=1 python \
    audiocap/train_whisper_supervised.py \
    --checkpoint-dir-root="../maratmp/audio-captioning/checkpoints" \
    --clotho-dir="../maratmp/audio-captioning/data/clotho_v2.1_backup/audiofolder" \
    --training-config="./configs/finetune_large_small-lr_no-augment_slice_clotho_backup_config.yaml" \
    --load-checkpoint="../maratmp/audio-captioning/checkpoints/atomic-sky-43/checkpoint-13500" \
    --wandb-group="finetuning"



    pretrain_3on1_tiny_config
    pretrain_1:12:3_large_half_scratch_config