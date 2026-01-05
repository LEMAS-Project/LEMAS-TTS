# LEMAS‑TTS: Multilingual Zero‑Shot TTS

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://www.python.org/downloads/release/python-3100/)
[![Report](https://img.shields.io/badge/GitHub-Report%20-red.svg)](https://lemas-project.github.io/LEMAS-Project/doc/report.pdf)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://lemas-project.github.io/LEMAS-Project/)
[![hfspace](https://img.shields.io/badge/🤗-Space%20Demo-yellow)](https://huggingface.co/spaces/LEMAS-Project/LEMAS-TTS)
[![hfmodel](https://img.shields.io/badge/🤗-Models%20Downloads-yellow)](https://huggingface.co/LEMAS-Project/LEMAS-TTS)

**LEMAS‑TTS** is a multilingual zero‑shot text‑to‑speech system, supporting 10 languages:
- Chinese
- English
- Spanish
- Russian
- French
- German
- Italian
- Portuguese
- Indonesian
- Vietnamese


## 1. Installation

### 1.1 Environment

```bash
git clone https://github.com/LEMAS-Project/LEMAS-TTS.git
cd ./LEMAS-TTS

# create a dedicated environment
conda create -n lemas-tts python=3.10
conda activate lemas-tts
```

### 1.2 System Dependencies

you can install the system dependencies as follows or via anaconda:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```
or
```bash
conda install -c conda-forge ffmpeg
```

### 1.3 Python Dependencies

```bash
pip install -r requirements.txt

# or, if you package it locally:
# pip install -e .
```
(Install PyTorch + Torchaudio according to your device (CUDA / ROCm / CPU / MPS),
following the official PyTorch instructions.)

### 1.4 Download Pretrained Models

Download the pretrained models from [https://huggingface.co/LEMAS-Project/LEMAS-TTS](https://huggingface.co/LEMAS-Project/LEMAS-TTS)

Then place the `pretrained_models/` folder next to the `lemas_tts/` package
root; the code locates the repo root by looking for this folder.


## 2. Usage

All commands below assume:

```bash
cd ./LEMAS-TTS
export PYTHONPATH="$PWD:${PYTHONPATH}"
```

### 2.1. Gradio Web UI (Multilingual Zero‑Shot TTS)

You can try the model via our Hugging Face space: [https://huggingface.co/spaces/LEMAS-Project/LEMAS-TTS](https://huggingface.co/spaces/LEMAS-Project/LEMAS-TTS)

Locally, you can run the Gradio web app with:

```bash
python lemas_tts/scripts/inference_gradio.py
```

The UI also exposes pre‑configured examples that use
`pretrained_models/demos/lemas_tts_test/*.wav` as reference audio.

You can customize host/port and sharing:

```bash
python lemas_tts/scripts/inference_gradio.py --host 0.0.0.0 --port 7860 --share
```

### 2.2. CLI: Multilingual TTS From Text

For simple TTS (text only, without reference audio), use:

- Python entry: `lemas_tts.scripts.tts_multilingual`
- Shell helper: `lemas_tts/scripts/tts_multilingual.sh`

Example:

```bash
cd ./LEMAS-TTS
bash lemas_tts/scripts/tts_multilingual.sh
```

The shell script demonstrates how to:

- Select `multilingual_grl` or `multilingual_prosody`
- Point to `pretrained_models/ckpts/...` and `pretrained_models/data/...`
- Choose frontend type (currently only support `phone`)
- Configure sampling parameters: NFE steps, CFG strength, Sway, speed, etc.

Or you can call the Python module directly, following the examples in bash scripts.

You can enable UVR5 denoising on the reference audio via `--denoise`.

### 2.3. CLI: Multilingual Speech Editing

For editing a region of an utterance given word‑level alignment JSONs, use:

- Python entry: `lemas_tts.scripts.speech_edit_multilingual`
- Shell helper: `lemas_tts/scripts/speech_edit_multilingual.sh`

The Python script expects:

- `--wav_dir`: directory with input `*.wav` files
- `--align_dir`: directory with Azure‑style alignment JSONs
- `--save_dir`: directory for edited outputs

Example:

```bash
cd ./LEMAS-TTS
bash lemas_tts/scripts/speech_edit_multilingual.sh
```

The script supports both prosody‑enabled and non‑prosody variants; see the
inline comments in `speech_edit_multilingual.sh` for a prosody example.


## 3. Acknowledgements

This project builds heavily on the following open‑source works:

- [F5‑TTS](https://github.com/SWivid/F5-TTS) – core model architecture and many
  components of the inference pipeline.
- [UVR5](https://github.com/Anjok07/ultimatevocalremovergui) – music source separation /
  vocal denoising, used here as an optional pre‑processing step.

If you use LEMAS‑TTS in your work, please also consider citing and acknowledging
these upstream projects.



## 4. Citation

```
@article{zhao2026lemas,
  title={LEMAS: A 150K-Hour Large-scale Extensible Multilingual Audio Suite with Generative Speech Models},
  author={Zhao, Zhiyuan and Lin, Lijian and Zhu, Ye and Xie, Kai and Liu, Yunfei and Li, Yu},
  year={2026}
}
```

## 5. License

This repository is released under the **CC‑BY‑NC‑4.0** license.  
See https://creativecommons.org/licenses/by-nc/4.0/ for more details.

