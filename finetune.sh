#!/usr/bin/env bash
python xlsr.py \
    --model_name_or_path="facebook/wav2vec2-large-xlsr-53" \
    --dataset_config_name="id" \
    --output_dir="wav2vec2-large-xlsr-indonesian" \
    --cache_dir="/workspace/cache" \
    --overwrite_output_dir \
    --num_train_epochs="60" \
    --per_device_eval_batch_size="32" \
    --per_device_train_batch_size="64" \
    --evaluation_strategy="steps" \
    --learning_rate="1e-4" \
    --warmup_steps="300" \
    --fp16 \
    --freeze_feature_extractor \
    --save_steps="100" \
    --eval_steps="100" \
    --save_total_limit="1" \
    --logging_steps="100" \
    --group_by_length \
    --feat_proj_dropout="0.04" \
    --layerdrop="0.041" \
    --attention_dropout="0.094" \
    --activation_dropout="0.055" \
    --hidden_dropout="0.047" \
    --mask_time_prob="0.4" \
    --do_train --do_eval \
    --gradient_accumulation_steps="1" \
    --dataloader_num_workers="8" \
    --push_to_hub True \
    --gradient_checkpointing \
    --report_to "wandb"
