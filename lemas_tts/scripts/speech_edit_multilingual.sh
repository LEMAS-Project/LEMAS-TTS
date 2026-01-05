#!/usr/bin/env bash

# Multilingual speech-editing wrapper for LEMAS-TTS.
# - Moves the example data from /mnt/outputs/... into pretrained_models (if present).
# - Runs lemas_tts.scripts.speech_edit_multilingual with the multilingual_grl model.

set -e

# Repo root
ROOT_DIR=$(pwd)
cd "$ROOT_DIR"

# Make sure local `lemas_tts` package is importable
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH}"

# New location under LEMAS-TTS pretrained_models
DATA_ROOT="$ROOT_DIR/pretrained_models/demos/lemas_edit_test"


# Example with prosody encoder (commented out by default)
CUDA_VISIBLE_DEVICES=0 python -m lemas_tts.scripts.speech_edit_multilingual \
  --wav_dir "${DATA_ROOT}/vocals" \
  --align_dir "${DATA_ROOT}/align" \
  --save_dir "${DATA_ROOT}/LEMAS_Edit_prosody" \
  --model multilingual_prosody \
  --ckpt_file pretrained_models/ckpts/multilingual_prosody/multilingual_prosody.safetensors \
  --vocab_file pretrained_models/data/multilingual_prosody/vocab.txt \
  --frontend phone \
  --use_ema \
  --enable_prosody_encoder \
  --prosody_cfg_path pretrained_models/ckpts/prosody_encoder/pretssel_cfg.json \
  --prosody_ckpt_path pretrained_models/ckpts/prosody_encoder/prosody_encoder_UnitY2.pt \
  --nfe_step 64 \
  --cfg_strength 5.0 \
  --sway_sampling_coef 3.0 \
  --ref_ratio 1.0 \
  --use_prosody_encoder \
  --seed -1

# Main example: multilingual_grl, no prosody encoder
CUDA_VISIBLE_DEVICES=0 python -m lemas_tts.scripts.speech_edit_multilingual \
  --wav_dir "${DATA_ROOT}/vocals" \
  --align_dir "${DATA_ROOT}/align" \
  --save_dir "${DATA_ROOT}/LEMAS_Edit_no_prosody" \
  --model multilingual_grl \
  --ckpt_file pretrained_models/ckpts/multilingual_grl/multilingual_grl.safetensors \
  --vocab_file pretrained_models/data/multilingual_grl/vocab.txt \
  --frontend phone \
  --speed 0.9 \
  --use_ema \
  --nfe_step 64 \
  --cfg_strength 5.0 \
  --sway_sampling_coef 3.0 \
  --ref_ratio 1.0 \
  --seed -1
