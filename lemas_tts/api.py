import os
import random
import sys
from pathlib import Path
import re, regex
import soundfile as sf
import tqdm
from hydra.utils import get_class
from omegaconf import OmegaConf

from lemas_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    transcribe,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)
from lemas_tts.model.utils import seed_everything
from lemas_tts.model.backbones.dit import DiT


# Resolve repository layout so we can find pretrained assets (ckpts, vocoder, etc.)
THIS_FILE = Path(__file__).resolve()
print("THIS_FILE:", THIS_FILE)

def _find_repo_root(start: Path) -> Path:
    """Locate the repo root by looking for a `pretrained_models` folder upwards."""
    for p in [start, *start.parents]:
        if (p / "pretrained_models").is_dir():
            return p
    cwd = Path.cwd()
    if (cwd / "pretrained_models").is_dir():
        return cwd
    return start


def _find_pretrained_root(start: Path) -> Path:
    """
    Locate the `pretrained_models` root, with support for:
    1) Explicit env override (LEMAS_PRETRAINED_ROOT)
    2) Hugging Face Spaces model mount under /models
    3) Local source tree (searching upwards from this file)
    """
    # 1) Explicit override
    env_root = os.environ.get("LEMAS_PRETRAINED_ROOT")
    if env_root:
        p = Path(env_root)
        if p.is_dir():
            return p

    # 2) HF Spaces model mount: /models/<model_id>/pretrained_models
    models_dir = Path("/models")
    if models_dir.is_dir():
        # Try the expected model name first
        specific = models_dir / "LEMAS-Project__LEMAS-TTS"
        if (specific / "pretrained_models").is_dir():
            return specific / "pretrained_models"
        # Otherwise, pick the first model that has a pretrained_models subdir
        for child in models_dir.iterdir():
            if child.is_dir() and (child / "pretrained_models").is_dir():
                return child / "pretrained_models"

    # 3) Local repo layout
    repo_root = _find_repo_root(start)
    if (repo_root / "pretrained_models").is_dir():
        return repo_root / "pretrained_models"

    cwd = Path.cwd()
    if (cwd / "pretrained_models").is_dir():
        return cwd / "pretrained_models"

    # Fallback: assume under repo root even if directory is missing
    return repo_root / "pretrained_models"


REPO_ROOT = _find_repo_root(THIS_FILE)
PRETRAINED_ROOT = _find_pretrained_root(THIS_FILE)
CKPTS_ROOT = PRETRAINED_ROOT / "ckpts"

class TTS:
    def __init__(
        self,
        model="multilingual",
        ckpt_file="",
        vocab_file="",
        ode_method="euler",
        use_ema=False,
        vocoder_local_path=str(CKPTS_ROOT / "vocos-mel-24khz"),
        use_prosody_encoder=False,
        prosody_cfg_path="",
        prosody_ckpt_path="",
        device=None,
        hf_cache_dir=None,
        frontend="phone",
    ):
        # Load model architecture config from bundled yaml
        config_dir = THIS_FILE.parent / "configs"
        model_cfg = OmegaConf.load(config_dir / f"{model}.yaml")
        # model_cls = get_class(f"lemas_tts.model.dit.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch

        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.target_sample_rate = model_cfg.model.mel_spec.target_sample_rate

        self.ode_method = ode_method
        self.use_ema = use_ema
        self.langs = {"cmn":"zh", "zh":"zh", "en":"en-us", "it":"it", "es":"es", "pt":"pt-br", "fr":"fr-fr", "de":"de", "ru":"ru", "id":"id", "vi":"vi", "th":"th"}
        
        if device is not None:
            self.device = device
        else:
            import torch

            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "xpu"
                if torch.xpu.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

        # # Load models
        # Prefer local vocoder directory if it exists; otherwise let `load_vocoder`
        # fall back to downloading from the default HF repo (charactr/vocos-mel-24khz).
        vocoder_is_local = False
        if vocoder_local_path is not None:
            try:
                vocoder_is_local = Path(vocoder_local_path).is_dir()
            except TypeError:
                vocoder_is_local = False

        self.vocoder = load_vocoder(
            self.mel_spec_type, vocoder_is_local, vocoder_local_path, self.device, hf_cache_dir
        )
        # self.vocoder = load_vocoder(vocoder_name="vocos", is_local=True, local_path=vocoder_local_path, device=self.device)
        if frontend is not None:
            from lemas_tts.infer.frontend import TextNorm
            # try:
                # Try requested frontend first (typically "phone")
            self.frontend = TextNorm(dtype=frontend)
            # except Exception as e:
            #     # If espeak/phonemizer is not available, gracefully fall back to char frontend
            #     print(f"[TTS] Failed to init TextNorm with dtype='{frontend}': {e}")
            #     print("[TTS] Falling back to char frontend (no espeak required).")
            #     self.frontend = TextNorm(dtype="char")
        else:
            self.frontend = None
        

        self.ema_model = load_model(
            DiT, model_arc, ckpt_file, self.mel_spec_type, vocab_file, self.ode_method, self.use_ema, self.device, 
            use_prosody_encoder=use_prosody_encoder, prosody_cfg_path=prosody_cfg_path, prosody_ckpt_path=prosody_ckpt_path,
        )

    def transcribe(self, ref_audio, language=None):
        return transcribe(ref_audio, language)

    def export_wav(self, wav, file_wave, remove_silence=False):
        sf.write(file_wave, wav, self.target_sample_rate)

        if remove_silence:
            remove_silence_for_generated_wav(file_wave)

    def export_spectrogram(self, spec, file_spec):
        save_spectrogram(spec, file_spec)

    def infer(
        self,
        ref_file,
        ref_text,
        gen_text,
        show_info=print,
        progress=tqdm,
        target_rms=0.1,
        cross_fade_duration=0.15,
        use_acc_grl=False,
        ref_ratio=None,
        no_ref_audio=False,
        cfg_strength=2,
        nfe_step=32,
        speed=1.0,
        sway_sampling_coef=5,
        separate_langs=False,
        fix_duration=None,
        use_prosody_encoder=True,
        file_wave=None,
        file_spec=None,
        seed=None,
    ):
        if seed is None:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        self.seed = seed

        ref_file, ref_text = preprocess_ref_audio_text(ref_file, ref_text)
        print("preprocesss:\n", "ref_file:", ref_file, "\nref_text:", ref_text)
        if self.frontend.dtype == "phone":
            ref_text = self.frontend.text2phn(ref_text+". ").replace("(cmn)", "(zh)").split("|")
            gen_text = gen_text.split("\n")
            gen_text = [self.frontend.text2phn(x+". ").replace("(cmn)", "(zh)").split("|") for x in gen_text]
        
        elif self.frontend.dtype == "char":
            src_lang, ref_text = self.frontend.text2norm(ref_text+". ")
            ref_text = ["("+src_lang.replace("cmn", "zh")+")"] + list(ref_text)
            gen_text = gen_text.split("\n")
            gen_text = [self.frontend.text2norm(x+". ") for x in gen_text]
            gen_text = [["("+x[0].replace("cmn", "zh")+")"] + list(x[1]) for x in gen_text]
        print("after frontend:\n", "ref_text:", ref_text, "\ngen_text:", gen_text)

        if separate_langs:
            ref_text = self.process_phone_list(ref_text) # Optional
            gen_text = [self.process_phone_list(x) for x in gen_text] 
        
        print("gen_text:", gen_text, "\nref_text:", ref_text)

        wav, sr, spec = infer_process(
            ref_file,
            ref_text,
            gen_text,
            self.ema_model,
            self.vocoder,
            self.mel_spec_type,
            show_info=show_info,
            progress=progress,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            use_prosody_encoder=use_prosody_encoder,
            use_acc_grl=use_acc_grl,
            ref_ratio=ref_ratio,
            no_ref_audio=no_ref_audio,
            speed=speed,
            fix_duration=fix_duration,
            device=self.device,
        )

        if file_wave is not None:
            self.export_wav(wav, file_wave, remove_silence=False)

        if file_spec is not None:
            self.export_spectrogram(spec, file_spec)

        return wav, sr, spec

    
    def process_phone_list(self, parts):
        puncs = {"#1", "#2", "#3", "#4", "_", "!", ",", ".", "?", '"', "'", "^", "。", "，", "？", "！"}
        """(vocab756 ver)处理phone list，给不带language id的phone添加当前language id前缀"""
        # parts = phn_str.split('|')
        processed = []
        current_lang = ""
        for i in range(len(parts)):
            part = parts[i]
            if part.startswith('(') and part.endswith(')') and part[1:-1] in self.langs:
                # 这是一个language id
                current_lang = part
                # processed.append(part)
            elif part in puncs: # not bool(regex.search(r'\p{L}', part[0])): # 匹配非字母数字、非空格的字符
                # 是停顿符或标点
                if len(processed) > 0 and processed[-1] == "_":
                    processed.pop()
                elif len(processed) > 0 and processed[-1] in puncs and part == "_":
                    continue
                processed.append(part)
                # if i < len(parts) - 1 and parts[i+1] != "_":
                #     processed.append("_")
            elif current_lang is not None:
                # 不是language id且有当前language id，添加前缀
                processed.append(f"{current_lang}{part}")
        return processed


if __name__ == "__main__":
    f5tts = F5TTS()

    wav, sr, spec = f5tts.infer(
        ref_file=str((THIS_FILE.parent / "infer" / "examples" / "basic" / "basic_ref_en.wav").resolve()),
        ref_text="some call me nature, others call me mother nature.",
        gen_text=(
            "I don't really care what you call me. I've been a silent spectator, watching species evolve, "
            "empires rise and fall. But always remember, I am mighty and enduring. Respect me and I'll nurture "
            "you; ignore me and you shall face the consequences."
        ),
        file_wave=str((REPO_ROOT / "outputs" / "api_out.wav").resolve()),
        file_spec=str((REPO_ROOT / "outputs" / "api_out.png").resolve()),
        seed=None,
    )

    print("seed :", f5tts.seed)
