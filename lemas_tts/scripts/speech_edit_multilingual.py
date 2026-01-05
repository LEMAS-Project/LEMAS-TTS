from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from lemas_tts.api import TTS


def load_wav_mono(path: str, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Load an audio file, convert to mono, resample, and clamp to [-0.999, 0.999]."""
    wav, sr = torchaudio.load(path)
    if wav.dim() > 1 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav.squeeze(), sr, target_sr).unsqueeze(0)
        sr = target_sr
    wav = torch.clip(wav, -0.999, 0.999)
    return wav.squeeze(0), sr


def build_tokens_from_text(tts: TTS, text: str) -> List[List[str]]:
    """
    Convert raw text into token sequence(s) consistent with the multilingual TTS training pipeline.

    We reuse the same frontend logic as in `TTS.infer`:
      - frontend.dtype == "phone" -> TextNorm.text2phn -> split on '|'
      - frontend.dtype == "char"  -> TextNorm.text2norm -> language tag + chars
      - frontend is None          -> simple character sequence as fallback
    """
    # Ensure sentence termination behaves similarly to inference_gradio / api
    text_proc = text.strip()
    if not text_proc.endswith((".", "。", "!", "？", "?", "！")):
        text_proc = text_proc + "."

    if getattr(tts, "frontend", None) is None:
        tokens = list(text_proc)
        return [tokens]

    dtype = getattr(tts.frontend, "dtype", "phone")

    if dtype == "phone":
        # TextNorm.text2phn returns a single string with '|' separators
        phones = tts.frontend.text2phn(text_proc + " ")
        phones = phones.replace("(cmn)", "(zh)")
        tokens = [tok for tok in phones.split("|") if tok]
        return [tokens]

    if dtype == "char":
        lang, norm = tts.frontend.text2norm(text_proc + " ")
        lang_tag = f"({lang.replace('cmn', 'zh')})"
        tokens = [lang_tag] + list(norm)
        return [tokens]

    # Fallback: character-level
    tokens = list(text_proc)
    return [tokens]


def gen_wav_multilingual(
    tts: TTS,
    segment_audio: torch.Tensor,
    sr: int,
    target_text: str,
    parts_to_edit: List[Tuple[float, float]],
    nfe_step: int = 64,
    cfg_strength: float = 5.0,
    sway_sampling_coef: float = 3.0,
    ref_ratio: float = 1.0,
    no_ref_audio: bool = False,
    use_acc_grl: bool = False,
    use_prosody_encoder_flag: bool = False,
    seed: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Core editing routine:
      - build an edit mask over the mel frames;
      - run CFM.sample with that mask and the new text;
      - decode mel to waveform via the vocoder.

    Args:
        tts:        TTS instance (provides ema_model, vocoder, frontend, device).
        segment_audio: 1D waveform tensor representing the utterance segment to edit (in seconds [0, T]).
        sr:         sampling rate of `segment_audio`.
        target_text: full text after editing.
        parts_to_edit: list of [start_sec, end_sec] to be replaced (relative to segment start).
    """
    device = tts.device
    model = tts.ema_model
    vocoder = tts.vocoder

    # Use model's own mel-spec configuration when possible
    mel_spec = getattr(model, "mel_spec", None)
    if mel_spec is None:
        raise RuntimeError("CFM model has no attached MelSpec; check your checkpoint.")

    target_sr = int(mel_spec.target_sample_rate)
    hop_length = int(mel_spec.hop_length)
    target_rms = 0.1

    if segment_audio.dim() == 1:
        audio = segment_audio.unsqueeze(0)
    else:
        audio = segment_audio

    # RMS normalization
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)

    audio = audio.to(device)

    # Build edit mask over mel frames
    offset = 0.0
    edit_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)
    for (start, end) in parts_to_edit:
        # small safety margin around the region to edit
        start = max(start - 0.1, 0.0)
        end = min(end + 0.1, audio.shape[-1] / target_sr)
        part_dur_sec = end - start
        part_dur_samples = int(round(part_dur_sec * target_sr))
        start_samples = int(round(start * target_sr))

        # frames before edited span: keep original (mask=True)
        num_keep_frames = int(round((start_samples - offset) / hop_length))
        # frames inside edited span: to be regenerated (mask=False)
        num_edit_frames = int(round(part_dur_samples / hop_length))

        if num_keep_frames > 0:
            edit_mask = torch.cat(
                [edit_mask, torch.ones(1, num_keep_frames, dtype=torch.bool, device=device)],
                dim=-1,
            )
        if num_edit_frames > 0:
            edit_mask = torch.cat(
                [edit_mask, torch.zeros(1, num_edit_frames, dtype=torch.bool, device=device)],
                dim=-1,
            )

        offset = end * target_sr

    # Pad mask to full sequence length (True = keep original)
    total_frames = audio.shape[-1] // hop_length
    if edit_mask.shape[-1] < total_frames + 1:
        pad_len = total_frames + 1 - edit_mask.shape[-1]
        edit_mask = F.pad(edit_mask, (0, pad_len), value=True)

    # Duration in frames
    duration = total_frames

    # Text tokens using multilingual frontend
    final_text_list = build_tokens_from_text(tts, target_text)

    # For multilingual models trained with `separate_langs=True`, we need to
    # post-process the phone sequence so that each non-punctuation token is
    # prefixed with its language id, consistent with training and the main API.
    # This mirrors the logic in `TTS.infer` where `process_phone_list` is
    # applied after text normalization.
    if hasattr(tts, "process_phone_list") and len(final_text_list) > 0:
        final_text_list = [tts.process_phone_list(final_text_list[0])]
    print("final_text_list:", final_text_list)
    # Sampling
    with torch.inference_mode():
        generated, _ = model.sample(
            cond=audio,
            text=final_text_list,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            seed=seed,
            edit_mask=edit_mask,
            use_acc_grl=use_acc_grl,
            use_prosody_encoder=use_prosody_encoder_flag,
            ref_ratio=ref_ratio,
            no_ref_audio=no_ref_audio,
        )

    # generated: [B, T_mel, C]
    generated = generated.to(torch.float32)
    generated_mel = generated.permute(0, 2, 1)  # [B, C, T_mel]

    # Vocoder decode (keep features on the same device as the vocoder/model)
    mel_for_vocoder = generated_mel.to(device)
    if tts.mel_spec_type == "vocos":
        wav_out = vocoder.decode(mel_for_vocoder)
    elif tts.mel_spec_type == "bigvgan":
        wav_out = vocoder(mel_for_vocoder)
    else:
        raise ValueError(f"Unsupported vocoder type: {tts.mel_spec_type}")

    if rms < target_rms:
        wav_out = wav_out * rms / target_rms

    return wav_out.squeeze(0), generated_mel


def run_edit_for_pair(
    tts: TTS,
    wav_path: str,
    json_path: str,
    save_path: str,
    *,
    nfe_step: int,
    cfg_strength: float,
    sway_sampling_coef: float,
    ref_ratio: float,
    no_ref_audio: bool,
    use_acc_grl: bool,
    use_prosody_encoder_flag: bool,
    seed: int | None,
) -> None:
    """Run speech editing for a single (wav, json) pair."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    wav, sr = load_wav_mono(wav_path, tts.target_sample_rate)
    data = json.loads(open(json_path, "r", encoding="utf-8").read())

    # Full utterance interval [sec]
    utt_start_sec, utt_end_sec = data["interval"]

    # Convert to sample indices
    utt_begin = int(round(utt_start_sec * sr))
    utt_end = int(round(utt_end_sec * sr))
    segment = wav[utt_begin:utt_end]

    # Determine edit span from modified_index
    start_idx, end_idx = data["modified_index"]
    words = data["words"]
    start_idx = max(0, start_idx)
    end_idx = min(len(words), end_idx)
    assert start_idx < end_idx, "modified_index range is empty."

    word_start_sec = words[start_idx]["interval"][0]
    word_end_sec = words[end_idx - 1]["interval"][1]

    # Relative to utterance segment
    edit_start_rel = max(0.0, word_start_sec - utt_start_sec - 0.1)
    edit_end_rel = min(word_end_sec - utt_start_sec, utt_end_sec - utt_start_sec + 0.1)

    parts_to_edit = [(edit_start_rel, edit_end_rel)]

    # Build target text by replacing the original phrase
    orig_phrase, new_phrase = data["modified_text"]
    display_text = data["display_text"]
    target_text = display_text.replace(orig_phrase, new_phrase)

    print(f"\n[EDIT] {os.path.basename(wav_path)}")
    print(f"  display_text : {display_text}")
    print(f"  modified_text: {orig_phrase!r} -> {new_phrase!r}")
    print(f"  target_text : {target_text}")
    print(f"  edit_span    : {parts_to_edit} (sec, relative to utterance)")

    start_t = time.time()
    gen_wav, _ = gen_wav_multilingual(
        tts=tts,
        segment_audio=segment,
        sr=sr,
        target_text=target_text,
        parts_to_edit=parts_to_edit,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        ref_ratio=ref_ratio,
        no_ref_audio=no_ref_audio,
        use_acc_grl=use_acc_grl,
        use_prosody_encoder_flag=use_prosody_encoder_flag,
        seed=seed,
    )
    elapsed = time.time() - start_t

    # torchaudio's ffmpeg backend requires CPU tensor input
    torchaudio.save(save_path, gen_wav.unsqueeze(0).cpu(), sr, format="wav")
    print(f"  saved: {save_path}  ({elapsed:.3f} s)")


def collect_pairs(
    wav: str | None,
    wav_dir: str,
    align_dir: str,
    save_dir: str,
) -> List[Tuple[str, str, str]]:
    """
    Build a list of (wav_path, json_path, save_path) triples.
    If `wav` is provided, only that file is used; otherwise, all .wav files in wav_dir.
    """
    pairs: List[Tuple[str, str, str]] = []

    if wav is not None:
        wav_paths = [wav]
    else:
        wav_paths = [
            os.path.join(wav_dir, f)
            for f in os.listdir(wav_dir)
            if f.lower().endswith(".wav") or f.lower().endswith(".mp3")
        ]
        wav_paths.sort()

    for wp in wav_paths:
        base = os.path.splitext(os.path.basename(wp))[0]
        jp = os.path.join(align_dir, base + ".json")
        sp = os.path.join(save_dir, base + ".wav")
        pairs.append((wp, jp, sp))

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Multilingual speech editing.")

    # Data paths
    parser.add_argument(
        "--wav",
        type=str,
        help="Path to a single input wav. If not set, use --wav_dir.",
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        help="Directory containing input wavs.",
    )
    parser.add_argument(
        "--align_dir",
        type=str,
        help="Directory containing Azure alignment JSONs.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory to save edited wavs.",
    )

    parser.add_argument("--model", type=str, default="multilingual")
    parser.add_argument("--ckpt_file", type=str, default="")
    parser.add_argument("--vocab_file", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--frontend",
        type=str,
        default="phone",
        choices=["phone", "char", "bpe", "none"],
    )
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument(
        "--enable_prosody_encoder",
        default=False,
        action="store_true",
        help="Build model with prosody encoder enabled.",
    )
    parser.add_argument("--prosody_cfg_path", type=str, default="")
    parser.add_argument("--prosody_ckpt_path", type=str, default="")

    # Inference hyperparameters (aligned with Gradio UI)
    parser.add_argument("--nfe_step", type=int, default=64)
    parser.add_argument("--speed", type=float, default=1.0)  # unused but kept for completeness
    parser.add_argument("--cfg_strength", type=float, default=5.0)
    parser.add_argument("--sway_sampling_coef", type=float, default=3.0)
    parser.add_argument("--ref_ratio", type=float, default=1.0)
    parser.add_argument("--no_ref_audio", action="store_true")
    parser.add_argument("--use_acc_grl", action="store_true")
    parser.add_argument(
        "--use_prosody_encoder",
        default=False,
        action="store_true",
        help="Use prosody encoder at inference (requires --enable_prosody_encoder).",
    )
    parser.add_argument("--seed", type=int, default=-1)

    args = parser.parse_args()

    if args.frontend == "none":
        frontend = None
    else:
        frontend = args.frontend

    tts = TTS(
        model=args.model,
        ckpt_file=args.ckpt_file,
        vocab_file=args.vocab_file,
        device=args.device,
        use_ema=args.use_ema,
        frontend=frontend,
        use_prosody_encoder=args.enable_prosody_encoder,
        prosody_cfg_path=args.prosody_cfg_path,
        prosody_ckpt_path=args.prosody_ckpt_path,
    )

    # Seed handling
    seed = None if args.seed == -1 else args.seed

    pairs = collect_pairs(
        wav=args.wav,
        wav_dir=args.wav_dir,
        align_dir=args.align_dir,
        save_dir=args.save_dir,
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    for wav_path, json_path, save_path in tqdm(pairs):
        if not os.path.exists(wav_path):
            print(f"[WARN] wav not found: {wav_path}")
            continue
        if not os.path.exists(json_path):
            print(f"[WARN] json not found: {json_path}")
            continue

        run_edit_for_pair(
            tts=tts,
            wav_path=wav_path,
            json_path=json_path,
            save_path=save_path,
            nfe_step=args.nfe_step,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_sampling_coef,
            ref_ratio=args.ref_ratio,
            no_ref_audio=args.no_ref_audio,
            use_acc_grl=args.use_acc_grl,
            use_prosody_encoder_flag=args.use_prosody_encoder and args.enable_prosody_encoder,
            seed=seed,
        )


if __name__ == "__main__":
    main()
