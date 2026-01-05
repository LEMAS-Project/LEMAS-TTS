"""
Command-line multilingual speech synthesis for LEMAS-TTS.

- Load a multilingual LEMAS-TTS checkpoint (multilingual_grl / multilingual_prosody).
- Optionally denoise the reference audio with UVR5.
- Run zero-shot TTS given reference audio + text and target text.
- Save the synthesized waveform to disk.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path

# Ensure repo root on path so `lemas_tts` imports work when run as a script.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lemas_tts.api import TTS, PRETRAINED_ROOT, CKPTS_ROOT  # noqa: E402


def _default_device() -> str:
    """Prefer CUDA if available, otherwise CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class UVR5:
    """Small wrapper around the bundled uvr5 implementation for denoising."""

    def __init__(self, model_dir: Path, code_dir: Path, device: str = "cpu") -> None:
        self.model_dir = str(model_dir)
        self.code_dir = str(code_dir)
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        import json

        if self.code_dir not in sys.path:
            sys.path.append(self.code_dir)

        from multiprocess_cuda_infer import ModelData, Inference  # type: ignore

        model_path = os.path.join(self.model_dir, "Kim_Vocal_1.onnx")
        config_path = os.path.join(self.model_dir, "MDX-Net-Kim-Vocal1.json")
        with open(config_path, "r", encoding="utf-8") as f:
            configs = json.load(f)
        model_data = ModelData(
            model_path=model_path,
            audio_path=self.model_dir,
            result_path=self.model_dir,
            device=self.device,
            process_method="MDX-Net",
            base_dir=self.model_dir,
            **configs,
        )

        uvr5_model = Inference(model_data, self.device)
        uvr5_model.load_model(model_path, 1)
        return uvr5_model

    def denoise_file(self, wav_path: str) -> str:
        """Denoise a wav file and return a path to a temporary denoised wav."""
        wav, sr = torchaudio.load(wav_path, channels_first=True)
        if wav.shape[0] == 1:
            wav = torch.cat((wav, wav), dim=0)  # mono -> stereo
        if sr != 44100:
            wav = torchaudio.functional.resample(wav.squeeze(), sr, 44100).unsqueeze(0)

        out = self.model.demix_base({0: wav.squeeze()}, is_match_mix=False, device=self.device)
        out = out.to("cpu").squeeze().numpy().T  # [T, 2]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, out, 44100, format="wav", subtype="PCM_24")
            return f.name


def _resolve_ckpt(model_name: str, ckpt_file: Optional[str]) -> str:
    """Resolve checkpoint path locally first, then from HF if needed."""
    if ckpt_file:
        return ckpt_file

    ckpt_dir = CKPTS_ROOT / model_name
    candidates = sorted(list(ckpt_dir.glob("*.safetensors")) + list(ckpt_dir.glob("*.pt")))
    if not candidates:
        # fallback: ckpts directly under CKPTS_ROOT
        root_candidates = sorted(
            list(CKPTS_ROOT.glob(f"{model_name}*.safetensors"))
            + list(CKPTS_ROOT.glob(f"{model_name}*.pt"))
        )
        candidates = root_candidates

    if candidates:
        return str(candidates[-1])

    # Remote mapping (matching inference_gradio.py / gradio_mix.py logic)
    HF_PRETRAINED_ROOT = "hf://LEMAS-Project/LEMAS-TTS/pretrained_models"
    remote_ckpts = {
        "multilingual_grl": f"{HF_PRETRAINED_ROOT}/ckpts/multilingual_grl/multilingual_grl.safetensors",
        "multilingual_prosody": f"{HF_PRETRAINED_ROOT}/ckpts/multilingual_prosody/multilingual_prosody.safetensors",
    }
    remote_path = remote_ckpts.get(model_name)
    if remote_path is None:
        raise FileNotFoundError(f"No ckpt found for model '{model_name}' under {CKPTS_ROOT}")
    resolved = cached_path(remote_path)
    return str(resolved)


def _resolve_vocab(model_name: str, vocab_file: Optional[str]) -> str:
    if vocab_file:
        return vocab_file
    vf = PRETRAINED_ROOT / "data" / model_name / "vocab.txt"
    if not vf.is_file():
        raise FileNotFoundError(f"Vocab file not found: {vf}")
    return str(vf)


def build_tts(
    model_name: str,
    ckpt_file: str,
    vocab_file: str,
    device: str,
    use_ema: bool,
    frontend: str,
    enable_prosody: bool,
    prosody_cfg_path: str = "",
    prosody_ckpt_path: str = "",
) -> TTS:
    # Prosody encoder config/ckpt: default under CKPTS_ROOT/prosody_encoder
    prosody_cfg = prosody_cfg_path or str(CKPTS_ROOT / "prosody_encoder" / "pretssel_cfg.json")
    prosody_ckpt = prosody_ckpt_path or str(CKPTS_ROOT / "prosody_encoder" / "prosody_encoder_UnitY2.pt")

    if enable_prosody:
        if not os.path.isfile(prosody_cfg) or not os.path.isfile(prosody_ckpt):
            raise FileNotFoundError(f"Prosody encoder assets not found: {prosody_cfg}, {prosody_ckpt}")

    # Decide whether to enable prosody by default based on model name.
    if model_name.endswith("grl"):
        use_prosody_encoder = False
    elif model_name.endswith("prosody"):
        use_prosody_encoder = enable_prosody
    else:
        use_prosody_encoder = enable_prosody

    return TTS(
        model=model_name,
        ckpt_file=ckpt_file,
        vocab_file=vocab_file,
        device=device,
        use_ema=use_ema,
        frontend=frontend,
        use_prosody_encoder=use_prosody_encoder,
        prosody_cfg_path=prosody_cfg if use_prosody_encoder else "",
        prosody_ckpt_path=prosody_ckpt if use_prosody_encoder else "",
    )


def main():
    parser = argparse.ArgumentParser(
        prog="python -m lemas_tts.scripts.speech_sysnthesis_multilingual",
        description="Multilingual zero-shot TTS with LEMAS-TTS.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="multilingual_grl",
        help="LEMAS-TTS model name (e.g. multilingual_grl, multilingual_prosody).",
    )
    parser.add_argument(
        "--ckpt_file",
        type=str,
        default="",
        help="Path to LEMAS-TTS checkpoint (.pt or .safetensors). "
        "If empty, will search under pretrained_models/ckpts or use HF remote.",
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        default="",
        help="Path to vocab.txt. If empty, will use pretrained_models/data/<model>/vocab.txt.",
    )
    parser.add_argument(
        "--frontend",
        type=str,
        default="phone",
        choices=["phone", "char"],
        help="Frontend type for TextNorm (phone/char).",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA weights from the checkpoint.",
    )
    parser.add_argument(
        "--enable_prosody_encoder",
        action="store_true",
        help="Enable prosody encoder if assets are available.",
    )
    parser.add_argument(
        "--prosody_cfg_path",
        type=str,
        default="",
        help="Optional path to prosody encoder config (pretssel_cfg.json).",
    )
    parser.add_argument(
        "--prosody_ckpt_path",
        type=str,
        default="",
        help="Optional path to prosody encoder checkpoint.",
    )

    parser.add_argument(
        "--ref_audio",
        type=str,
        required=True,
        help="Reference audio wav path.",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        required=True,
        help="Reference transcript text.",
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Generation text. Use '\\n' for multiple sentences.",
    )
    parser.add_argument(
        "--output_wave",
        type=str,
        default="output.wav",
        help="Output wav path.",
    )

    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply UVR5 denoising to the reference audio before TTS.",
    )

    # Sampling / guidance parameters
    parser.add_argument("--nfe_step", type=int, default=64, help="Number of sampling steps (NFE).")
    parser.add_argument("--cfg_strength", type=float, default=5.0, help="CFG strength.")
    parser.add_argument(
        "--sway_sampling_coef",
        type=float,
        default=3.0,
        help="Sway sampling coefficient.",
    )
    parser.add_argument(
        "--ref_ratio",
        type=float,
        default=1.0,
        help="How much to rely on reference audio.",
    )
    parser.add_argument(
        "--no_ref_audio",
        action="store_true",
        help="Disable reference audio conditioning.",
    )
    parser.add_argument(
        "--separate_langs",
        action="store_true",
        help="Apply language tags per token (for multilingual models).",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Generation speed factor.",
    )
    parser.add_argument(
        "--use_acc_grl",
        action="store_true",
        help="Use accent GRL conditioning (if the model supports it).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed; -1 for random.",
    )

    args = parser.parse_args()

    # Resolve checkpoint & vocab
    ckpt_file = _resolve_ckpt(args.model, args.ckpt_file or None)
    vocab_file = _resolve_vocab(args.model, args.vocab_file or None)

    # Ref audio (optional denoise)
    ref_audio_path = args.ref_audio
    if not os.path.isfile(ref_audio_path):
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    tmp_denoised = None
    if args.denoise:
        uvr_root = PRETRAINED_ROOT / "uvr5"
        uvr_code = REPO_ROOT / "uvr5"
        uv = UVR5(model_dir=uvr_root, code_dir=uvr_code, device="cpu")
        tmp_denoised = uv.denoise_file(ref_audio_path)
        ref_audio_path = tmp_denoised

    # Build TTS (prefer CUDA, fall back to CPU if incompatible)
    preferred_device = _default_device()

    def _build(device_for_tts: str) -> TTS:
        return build_tts(
            model_name=args.model,
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            device=device_for_tts,
            use_ema=args.use_ema,
            frontend=args.frontend,
            enable_prosody=args.enable_prosody_encoder,
            prosody_cfg_path=args.prosody_cfg_path,
            prosody_ckpt_path=args.prosody_ckpt_path,
        )

    try:
        tts = _build(preferred_device)
    except Exception:
        # If CUDA initialization fails, try CPU.
        tts = _build("cpu")

    # Run inference
    seed = None if args.seed == -1 else args.seed

    try:
        tts.infer(
            ref_file=ref_audio_path,
            ref_text=args.ref_text.strip(),
            gen_text=args.text.strip(),
            nfe_step=int(args.nfe_step),
            cfg_strength=float(args.cfg_strength),
            sway_sampling_coef=float(args.sway_sampling_coef),
            use_acc_grl=bool(args.use_acc_grl),
            ref_ratio=float(args.ref_ratio),
            no_ref_audio=bool(args.no_ref_audio),
            separate_langs=bool(args.separate_langs),
            speed=float(args.speed),
            use_prosody_encoder=args.enable_prosody_encoder,
            file_wave=args.output_wave,
            seed=seed,
        )
        print(f"Saved synthesized audio to: {args.output_wave}")
    finally:
        if tmp_denoised is not None and os.path.isfile(tmp_denoised):
            os.remove(tmp_denoised)


if __name__ == "__main__":
    main()

