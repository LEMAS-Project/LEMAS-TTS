"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from random import random
import random as _random
from typing import Callable, Dict, OrderedDict
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from lemas_tts.model.modules import MelSpec
from lemas_tts.model.modules import MIEsitmator, AccentClassifier, grad_reverse
from lemas_tts.model.backbones.ecapa_tdnn import ECAPA_TDNN
from lemas_tts.model.backbones.prosody_encoder import ProsodyEncoder, extract_fbank_16k
from lemas_tts.model.utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


def clip_and_shuffle(mel, mel_len, sample_rate=24000, hop_length=256, ratio=None):
    """
    Randomly clip a mel-spectrogram segment and shuffle 1-second chunks to
    create an accent-invariant conditioning segment.

    This is a inference-time utility used by the accent GRL path.

    Args:
        mel: [n_mels, T]
        mel_len: int, original mel length (T)
    """
    frames_per_second = int(sample_rate / hop_length)  # ≈ 94 frames / second

    # ---- 1. Randomly crop 25%~75% of the original length (or ratio * length) ----
    total_len = mel_len
    if not ratio:
        seg_len = _random.randint(int(0.25 * total_len), int(0.75 * total_len))
    else:
        seg_len = int(total_len * ratio)
    start = _random.randint(0, max(0, total_len - seg_len))
    mel_seg = mel[:, start : start + seg_len]

    # ---- 2. Split into ~1-second chunks ----
    n_chunks = (mel_seg.size(1) + frames_per_second - 1) // frames_per_second
    chunks = []
    for i in range(n_chunks):
        chunk = mel_seg[:, i * frames_per_second : (i + 1) * frames_per_second]
        chunks.append(chunk)

    # ---- 3. Shuffle chunk order ----
    _random.shuffle(chunks)
    shuffled_mel = torch.cat(chunks, dim=1)

    # ---- 4. Repeat random chunks until reaching original length ----
    if shuffled_mel.size(1) < total_len:
        repeat_chunks = []
        while sum(c.size(1) for c in repeat_chunks) < total_len:
            repeat_chunks.append(_random.choice(chunks))
        shuffled_mel = torch.cat([shuffled_mel] + repeat_chunks, dim=1)

    # ---- 5. Trim to exactly mel_len ----
    shuffled_mel = shuffled_mel[:, :total_len]
    assert shuffled_mel.shape == mel.shape, f"shuffled_mel.shape != mel.shape: {shuffled_mel.shape} != {mel.shape}"

    return shuffled_mel

class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        text_drop_prob=0.1,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
        use_ctc_loss: bool = False,
        use_spk_enc: bool = False,
        use_prosody_encoder: bool = False,
        prosody_cfg_path: str | None = None,
        prosody_ckpt_path: str | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.text_drop_prob = text_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

        # Prosody encoder (Pretssel ECAPA-TDNN)
        self.use_prosody_encoder = (
            use_prosody_encoder and prosody_cfg_path is not None and prosody_ckpt_path is not None
        )
        if self.use_prosody_encoder:
            cfg_path = Path(prosody_cfg_path)
            ckpt_path = Path(prosody_ckpt_path)
            self.prosody_encoder = ProsodyEncoder(cfg_path, ckpt_path, freeze=True)
            # 512-d prosody -> mel channel dimension
            self.prosody_to_mel = nn.Linear(512, self.num_channels)
            self.prosody_dropout = nn.Dropout(p=0.2)
        else:
            self.prosody_encoder = None
        
        # Speaker encoder
        self.use_spk_enc = use_spk_enc
        if use_spk_enc:
            self.speaker_encoder = ECAPA_TDNN(
                self.num_channels,
                self.dim,
                channels=[512, 512, 512, 512, 1536],
                kernel_sizes=[5, 3, 3, 3, 1],
                dilations=[1, 2, 3, 4, 1],
                attention_channels=128,
                res2net_scale=4,
                se_channels=128,
                global_context=True,
                batch_norm=True,
            )
            # self.load_partial_weights(self.speaker_encoder, "/cto_labs/vistring/zhaozhiyuan/outputs/F5-TTS/pretrain/speaker.bin", device="cpu")

        self.use_ctc_loss = use_ctc_loss
        if use_ctc_loss:
            # print("vocab_char_map:", len(vocab_char_map)+1, "dim:", dim, "mel_spec_kwargs:",mel_spec_kwargs)
            self.ctc = MIEsitmator(len(self.vocab_char_map), self.num_channels, self.dim, dropout=self.text_drop_prob)

        self.accent_classifier = AccentClassifier(input_dim=self.num_channels, hidden_dim=self.dim, num_accents=12)
        self.accent_criterion = nn.CrossEntropyLoss()

    def load_partial_weights(self, model: nn.Module,
                            ckpt_path: str,
                            device="cpu",
                            verbose=True) -> int:
        """
        仅加载形状匹配的参数，其余跳过。
        返回成功加载的参数数量。
        """
        state_dict = torch.load(ckpt_path, map_location=device)
        model_dict = model.state_dict()

        ok_count = 0
        new_dict: OrderedDict[str, torch.Tensor] = OrderedDict()

        for k, v in state_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                new_dict[k] = v
                ok_count += 1
            else:
                if verbose:
                    print(f"[SKIP] {k}  ckpt:{v.shape}  model:{model_dict[k].shape if k in model_dict else 'N/A'}")

        model_dict.update(new_dict)
        model.load_state_dict(model_dict)
        if verbose:
            print(f"=> 成功加载 {ok_count}/{len(state_dict)} 个参数")
        return ok_count

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        use_acc_grl = True,
        use_prosody_encoder = True,
        ref_ratio = 1,
    ):
        self.eval()

        # raw wave -> mel, keep a copy for prosody encoder if available
        raw_audio = None
        if cond.ndim == 2:
            raw_audio = cond.clone()  # (B, nw)
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)
        cond_mean = cond.mean(dim=1, keepdim=True)
        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # optional global prosody conditioning at inference (one embedding per sample)
        prosody_mel_cond = None
        prosody_text_cond = None
        prosody_embeds = None
        if self.prosody_encoder is not None and raw_audio is not None and use_prosody_encoder:
            embeds = []
            for b in range(batch):
                audio_b = raw_audio[b].unsqueeze(0)  # (1, nw)
                src_sr = self.mel_spec.target_sample_rate
                if src_sr != 16_000:
                    audio_16k = torchaudio.functional.resample(
                        audio_b, src_sr, 16_000
                    ).squeeze(0)
                else:
                    audio_16k = audio_b.squeeze(0)
                fbank = extract_fbank_16k(audio_16k)
                fbank = fbank.unsqueeze(0).to(device=device, dtype=cond.dtype)
                emb = self.prosody_encoder(fbank, padding_mask=None)[0]  # (512,)
                embeds.append(emb)
            prosody_embeds = torch.stack(embeds, dim=0)  # (B, 512)
            # broadcast along mel and text
            prosody_mel_cond = prosody_embeds[:, None, :].expand(-1, cond_seq_len, -1)

        if use_acc_grl:
            # rand_mel = clip_and_shuffle(cond.permute(0, 2, 1).squeeze(0), cond.shape[1])
            # rand_mel = rand_mel.unsqueeze(0).permute(0, 2, 1)
            # assert rand_mel.shape == cond.shape, f"Shape diff: rand_mel.shape: {rand_mel.shape}, cond.shape: {cond.shape}"
            # cond_grl = grad_reverse(rand_mel, lambda_=1.0)

            if ref_ratio < 1:
                rand_mel = clip_and_shuffle(cond.permute(0, 2, 1).squeeze(0), cond.shape[1], ratio=ref_ratio)
                rand_mel = rand_mel.unsqueeze(0).permute(0, 2, 1)
                assert rand_mel.shape == cond.shape, f"Shape diff: rand_mel.shape: {rand_mel.shape}, cond.shape: {cond.shape}"
                cond_grl = grad_reverse(rand_mel, lambda_=1.0)
            else:
                cond_grl = grad_reverse(cond, lambda_=1.0)
            # print("cond:", cond.shape, cond.mean(), cond.max(), cond.min(), "rand_mel:", rand_mel.mean(), rand_mel.max(), rand_mel.min(), "cond_grl:", cond_grl.mean(), cond_grl.max(), cond_grl.min())

        # text

        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # duration

        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(
            torch.maximum((text != -1).sum(dim=-1), lens) + 1, duration
        )  # duration at least text/audio prompt length plus one token, so something is generated
        # clamp and convert max_duration to python int for padding ops
        duration = duration.clamp(max=max_duration)
        max_duration = int(duration.amax().item())

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)

        if prosody_mel_cond is not None:
            prosody_mel_cond = F.pad(
                prosody_mel_cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0
            )
            prosody_mel_proj = self.prosody_to_mel(prosody_mel_cond)
            cond = cond + prosody_mel_proj

        if no_ref_audio:
            random_cond = torch.randn_like(cond) * 0.1 + cond_mean
            random_cond = random_cond / random_cond.mean(dim=1, keepdim=True) * cond_mean
            print("cond:", cond.mean(), cond.max(), cond.min(), "random_cond:", random_cond.mean(), random_cond.max(), random_cond.min(), "mean_cond:", cond_mean.shape)
            cond = random_cond
        
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)

        if use_acc_grl:
            cond_grl = F.pad(cond_grl, (0, 0, 0, max_duration - cond_seq_len), value=0.0)


        step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))  # allow direct control (cut cond audio) with lens passed in
        

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # neural ode

        def compute_sway_max(steps: int,
                            t_start: float = 0.0,
                            dtype=torch.float32,
                            min_ratio: float | None = None,
                            safety_factor: float = 0.5) -> float:
            """
            Compute a safe upper bound for sway_sampling_coef given steps and t_start.

            - steps: number of ODE steps
            - t_start: start time in [0,1)
            - dtype: torch dtype (for machine eps)
            - min_ratio: smallest distinguishable dt^p (if None, use conservative default)
            - safety_factor: scale down the theoretical maximum to be safe
            """
            assert 0.0 <= t_start < 1.0
            dt = (1.0 - t_start) / max(1, steps)
            eps = torch.finfo(dtype).eps

            if min_ratio is None:
                # conservative default: ~100 * eps (float32 -> ~1e-5)
                min_ratio = max(1e-9, 1e2 * float(eps))

            if dt >= 0.9:
                p_max = 1.0 + 10.0
            else:
                # solve dt^p >= min_ratio  =>  p <= log(min_ratio)/log(dt)
                p_max = math.log(min_ratio) / math.log(dt)

            sway_max = max(0.0, p_max - 1.0)
            sway_max = sway_max * float(safety_factor)
            return torch.tensor(sway_max, device=device, dtype=dtype)

        # prepare text-side prosody conditioning if embeddings available
        if prosody_embeds is not None:
            text_len = text.shape[1]
            prosody_text_cond = prosody_embeds[:, None, :].expand(-1, text_len, -1)
        else:
            prosody_text_cond = None

        def fn(t, x):
            # at each step, conditioning is fixed
            # if use_spk_enc:
            #     mix_cond = t * cond + (1-t) * spk_emb
            #     step_cond = torch.where(cond_mask, mix_cond, torch.zeros_like(mix_cond))
            if use_acc_grl:
                step_cond = torch.where(cond_mask, cond_grl, torch.zeros_like(cond_grl))
            else:
                step_cond = torch.where(cond_mask, cond, torch.zeros_like(cond))
            
            # predict flow
            pred = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=False,
                drop_text=False,
                cache=True,
                prosody_text=prosody_text_cond,
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x,
                cond=step_cond,
                text=text,
                time=t,
                mask=mask,
                drop_audio_cond=True,
                drop_text=True,
                cache=True,
                prosody_text=prosody_text_cond,
            )
            # cfg_t = cfg_strength * torch.cos(0.5 * torch.pi * t)
            # cfg_t = cfg_strength * (1 - t)
            cfg_t = cfg_strength * ((1 - t) ** 2)
            # print("t:", t, "cfg_t:", cfg_t)
            res = pred + (pred - null_pred) * cfg_t
            # print("t:", t.item(), "\tres:", res.shape, res.mean().item(), res.max().item(), res.min().item(), "\tpred:", pred.mean().item(), pred.max().item(), pred.min().item(), "\tnull_pred:", null_pred.mean().item(), null_pred.max().item(), null_pred.min().item(), "\tcfg_t:", cfg_t.item())
            res = res.clamp(-20, 20)
            return res

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1, int(steps + 1), device=self.device, dtype=step_cond.dtype)

        sway_max = compute_sway_max(steps, t_start=t_start, dtype=step_cond.dtype, min_ratio=1e-9, safety_factor=0.7)
        if sway_sampling_coef is not None:
            sway_sampling_coef = min(sway_max, sway_sampling_coef)
            # t = t + sway_sampling_coef *  (torch.cos(torch.pi / 2 * t) - 1 + t)
            t = t ** (1 + sway_sampling_coef)
        else:
            t = t ** (1 + sway_max)
        # print("t:",t, "sway_max:", sway_max, "sway_sampling_coef:", sway_sampling_coef)
    
        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        self.transformer.clear_cache()

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)

        # out生成的部分，或者说pad补0的部分，单独计算mean, 然后和cond的mean做对齐（乘以系数，两个的均值要差不多）
        if no_ref_audio:
            out_mean = out[:,cond_seq_len:,:].mean(dim=1, keepdim=True)
            out[:,cond_seq_len:,:] = out[:,cond_seq_len:,:] - (out_mean - cond_mean) 
            # print("out_mean:", out_mean.shape, out_mean.mean(), "cond_mean:", cond_mean.shape, cond_mean.mean(), "out:", out[:,cond_seq_len:,:].shape, out[:,cond_seq_len:,:].mean().item(), out[:,cond_seq_len:,:].max().item(), out[:,cond_seq_len:,:].min().item())

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)
        # print("out:", out.shape, "trajectory:", trajectory.shape)
        return out, trajectory


    def info_nce_speaker(self, 
                        e_gt: torch.Tensor,
                        e_pred: torch.Tensor,
                        temperature: float = 0.1):
        """
        InfoNCE loss for speaker encoder training.
        同一条样本的 e_gt 与 e_pred 互为正例，其余均为负例。

        Args:
            temperature: 温度缩放 τ

        Returns:
            loss: 标量 tensor，可 backward
        """
        B = e_gt.size(0)
        # 2. L2 归一化
        e_gt   = F.normalize(e_gt,   dim=1)
        e_pred = F.normalize(e_pred, dim=1)

        # 3. 计算 B×B 相似度矩阵（pred 对 gt）
        logits = torch.einsum('bd,cd->bc', e_pred, e_gt) / temperature  # [B, B]

        # 4. 正例标签正好是对角线
        labels = torch.arange(B, device=logits.device)

        # 5. InfoNCE = cross-entropy over in-batch negatives
        loss = F.cross_entropy(logits, labels)
        return loss


    def forward(self, batchs: Dict[str, torch.Tensor], *, noise_scheduler: str | None = None):
        """
        Simplified forward version for accent-invariant flow matching.
        Removes speaker encoder and CTC parts, keeps accent GRL.
        """
        inp = batchs["mel"].permute(0, 2, 1)           # [B, T_mel, D]
        lens = batchs["mel_lengths"]
        text = batchs["text"]
        langs = batchs["langs"]
        audio_16k_list = batchs.get("audio_16k", None)
        prosody_idx_list = batchs.get("prosody_idx", None)

        # # ---- 4. 随机截取并打乱 segment ----
        # rand_mel = [clip_and_shuffle(spec, spec.shape[-1]) for spec in batchs["mel"]]
        
        # padded_rand_mel = []
        # for spec in rand_mel:
        #     padding = (0, batchs["mel"].shape[-1] - spec.size(-1))
        #     padded_spec = F.pad(spec, padding, value=0)
        #     padded_rand_mel.append(padded_spec)
        # rand_mel = torch.stack(padded_rand_mel).permute(0, 2, 1)       
        # assert rand_mel.shape == inp.shape, f"shape diff: rand_mel.shape: {rand_mel.shape}, inp.shape: {inp.shape}"

        if inp.ndim == 2:
            inp = self.mel_spec(inp).permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device = *inp.shape[:2], inp.dtype, self.device

        # --- handle text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch
        # print("text:", batchs["text"][0], text.shape, text[0], batchs["text_lengths"][0])
        # --- prosody conditioning (compute embeddings per sub-utterance)
        prosody_mel_cond = None
        prosody_text_cond = None
        if (
            self.prosody_encoder is not None
            and audio_16k_list is not None
            and prosody_idx_list is not None
        ):
            # prepare zero tensors for each sample
            T_mel = seq_len
            T_text = text.shape[1]
            prosody_mel_cond = torch.zeros(batch, T_mel, 512, device=device, dtype=dtype)
            prosody_text_cond = torch.zeros(batch, T_text, 512, device=device, dtype=dtype)

            # collect all segments, run encoder per segment
            seg_embeds: list[Tensor] = []
            seg_meta: list[tuple[int, int, int, int, int, int]] = []
            for b in range(batch):
                audio_b = audio_16k_list[b]
                idx_list = prosody_idx_list[b]
                if audio_b is None or idx_list is None:
                    continue
                audio_b = audio_b.to(device=device, dtype=dtype)
                for seg in idx_list:
                    text_start, text_end, mel_start, mel_end, audio_start, audio_end = seg
                    # clamp audio indices
                    audio_start = max(0, min(audio_start, audio_b.shape[0] - 1))
                    audio_end = max(audio_start + 1, min(audio_end, audio_b.shape[0]))
                    audio_seg = audio_b[audio_start:audio_end]
                    if audio_seg.numel() == 0:
                        continue
                    fbank = extract_fbank_16k(audio_seg)  # (T_fbank, 80)
                    fbank = fbank.unsqueeze(0).to(device=device, dtype=dtype)  # (1, T_fbank, 80)
                    with torch.no_grad():
                        emb = self.prosody_encoder(fbank, padding_mask=None)[0]  # (512,)
                    seg_embeds.append(emb)
                    seg_meta.append(
                        (b, text_start, text_end, mel_start, mel_end)
                    )

            if seg_embeds:
                seg_embeds_tensor = torch.stack(seg_embeds, dim=0)  # (N_seg, 512)
                # scatter embeddings back to per-sample tensors
                for emb, meta in zip(seg_embeds_tensor, seg_meta):
                    b, ts, te, ms, me = meta
                    emb_exp = emb.to(device=device, dtype=dtype)
                    prosody_mel_cond[b, ms:me, :] = emb_exp
                    prosody_text_cond[b, ts:te, :] = emb_exp

            # dropout on prosody conditioning
            prosody_mel_cond = self.prosody_dropout(prosody_mel_cond)
            prosody_text_cond = self.prosody_dropout(prosody_text_cond)

        # --- mask & random span
        mask = lens_to_mask(lens, length=seq_len)
        frac_lengths = torch.zeros((batch,), device=device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
        if exists(mask):
            rand_span_mask &= mask

        # --- flow setup
        x1 = inp
        x0 = torch.randn_like(x1)
        time = torch.rand((batch,), dtype=dtype, device=device)
        t = time[:, None, None]
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # --- conditional input (masked mel) + optional prosody
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1) # x1 # rand_mel
        if prosody_mel_cond is not None:
            prosody_mel_proj = self.prosody_to_mel(prosody_mel_cond)  # (B, T_mel, num_channels)
            # if needed, pad/crop to seq_len
            if prosody_mel_proj.size(1) < seq_len:
                pad_len = seq_len - prosody_mel_proj.size(1)
                prosody_mel_proj = F.pad(prosody_mel_proj, (0, 0, 0, pad_len))
            elif prosody_mel_proj.size(1) > seq_len:
                prosody_mel_proj = prosody_mel_proj[:, :seq_len, :]
            cond = cond + prosody_mel_proj
        
        # --- Gradient reversal: encourage accent-invariant cond
        cond_grl = grad_reverse(cond, lambda_=1.0)

        # # --- random drop condition for CFG-like robustness
        # drop_audio_cond = random() < self.audio_drop_prob
        # drop_text_cond = random() < self.text_drop_prob if not drop_audio_cond else True

        # safe per-batch random (tensor)
        rand_for_drop = torch.rand(1, device=device)
        drop_audio_cond = (rand_for_drop.item() < self.audio_drop_prob)
        rand_for_text = torch.rand(1, device=device)
        drop_text_cond = (rand_for_text.item() < self.text_drop_prob)

        # --- main prediction
        pred = self.transformer(
            x=φ,
            cond=cond_grl,
            text=text,
            time=time,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text_cond,
            prosody_text=prosody_text_cond,
        )

        # === FLOW LOSS (robust mask-weighted) ===
        pred_clamp = pred.float().clamp(-20, 20)
        per_elem_loss = F.mse_loss(pred_clamp, flow, reduction="none")  # [B, T, D]

        mask_exp = rand_span_mask.unsqueeze(-1).to(dtype=per_elem_loss.dtype)  # [B, T, 1]
        masked_loss = per_elem_loss * mask_exp  # zeros where mask False

        # total selected scalar (frames * dim)
        n_selected = mask_exp.sum() * per_elem_loss.size(-1)  # scalar
        denom = torch.clamp(n_selected, min=1.0)

        loss_sum = masked_loss.sum()
        loss = loss_sum / denom
        # numeric safety
        loss = torch.where(torch.isnan(loss) | (loss > 300.0), torch.tensor(300.0, device=loss.device, dtype=loss.dtype), loss)

        # === ACCENT LOSS ===
        accent_logits = self.accent_classifier(cond_grl)
        # pool across time -> [B, C]
        accent_logits_mean = accent_logits.mean(dim=1)
        lang_labels = langs.to(accent_logits_mean.device).long()
        accent_loss = self.accent_criterion(accent_logits_mean, lang_labels)
        # guard against NaN / Inf in accent_loss
        if not torch.isfinite(accent_loss):
            accent_loss = torch.zeros_like(accent_loss, device=accent_loss.device)

        base_loss = loss + 0.1 * accent_loss

        # === OPTIONAL CTC LOSS (robust, only on valid samples) ===
        ctc_scaled = torch.tensor(0.0, device=device, dtype=dtype)
        if getattr(self, "use_ctc_loss", False) and getattr(self, "ctc", None) is not None:
            # select samples with larger t for CTC supervision (similar to forward_old)
            valid_indices = torch.where(time > 0.5)[0]
            if valid_indices.size(0) > 2:
                selected_pred = pred[valid_indices]
                selected_text = text[valid_indices]
                selected_lens = lens[valid_indices]
                # text was tokenized from list_str_to_idx, where padding is -1
                selected_target_lengths = (selected_text != -1).sum(dim=-1)

                ctc_loss = self.ctc(
                    decoder_outputs=selected_pred,
                    target_phones=selected_text,
                    decoder_lengths=selected_lens,
                    target_lengths=selected_target_lengths,
                )
                if torch.isfinite(ctc_loss) and ctc_loss.item() > 1e-6:
                    ctc_scaled = ctc_loss
                    base_loss = base_loss + 0.1 * ctc_scaled

        total_loss = base_loss

        # note: we intentionally do NOT add 0.0 * pred.sum() etc. here, to avoid
        # propagating NaNs from intermediate tensors into the loss scalar.

        return total_loss, accent_loss, ctc_scaled, cond, pred
