#!/usr/bin/env bash

# Simple wrapper around the multilingual LEMAS-TTS CLI.
# Adjust paths and arguments as needed.

set -e

# Repo root
ROOT_DIR=$(pwd)
cd "$ROOT_DIR"

# Make sure local `lemas_tts` package is importable
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH}"

# Example 1: multilingual_grl, Spanish reference -> Chinese target
python -m lemas_tts.scripts.tts_multilingual \
  --model multilingual_grl \
  --ckpt_file pretrained_models/ckpts/multilingual_grl/multilingual_grl.safetensors \
  --vocab_file pretrained_models/data/multilingual_grl/vocab.txt \
  --frontend phone \
  --use_ema \
  --ref_audio pretrained_models/demos/lemas_tts_test/es.wav \
  --ref_text "Te voy a dar un tip que le copié a John Rockefeller, uno de los empresarios más picudos de la historia. " \
  --text "我要给你一个从历史上最精明的商人之一约翰·洛克菲勒那里抄来的秘诀。" \
  --output_wave pretrained_models/demos/lemas_tts_test/tts_es_zh.wav \
  --nfe_step 64 \
  --cfg_strength 5.0 \
  --sway_sampling_coef 3.0 \
  --ref_ratio 1.0 \
  --separate_langs \
  --speed 1.0 \
  --seed -1

# Example 2: multilingual_grl, Portuguese reference -> English target, with denoise
python -m lemas_tts.scripts.tts_multilingual \
  --model multilingual_grl \
  --ckpt_file pretrained_models/ckpts/multilingual_grl/multilingual_grl.safetensors \
  --vocab_file pretrained_models/data/multilingual_grl/vocab.txt \
  --frontend phone \
  --use_ema \
  --ref_audio pretrained_models/demos/lemas_tts_test/pt.wav \
  --ref_text "Nova, dia 25 desse mês vai rolar operação the last Frontier. " \
  --text "Preparations are currently underway to ensure the operation proceeds as planned. " \
  --output_wave pretrained_models/demos/lemas_tts_test/tts_pt_en.wav \
  --nfe_step 64 \
  --cfg_strength 5.0 \
  --sway_sampling_coef 3.0 \
  --ref_ratio 1.0 \
  --separate_langs \
  --speed 1.0 \
  --denoise \
  --seed -1

