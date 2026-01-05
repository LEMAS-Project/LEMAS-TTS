"""
Prosody encoder backbone based on the Pretssel ECAPA-TDNN architecture.

This module provides:
  - ProsodyEncoder: wraps an ECAPA-TDNN model to produce utterance-level
    prosody embeddings from 80-dim FBANK features.
  - extract_fbank_16k: utility to compute 80-bin FBANK from 16kHz audio.

It is self-contained (no fairseq2 dependency) and can be used inside
CFM or other models as a conditioning network.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import json

import torch
import torchaudio
from torch import Tensor
from torch import nn
from torch.nn import Conv1d, LayerNorm, Module, ModuleList, ReLU, Sigmoid, Tanh, init
import torch.nn.functional as F


AUDIO_SAMPLE_RATE = 16_000


class ECAPA_TDNN(Module):
    """
    ECAPA-TDNN core used in Pretssel prosody encoder.

    Expects input features of shape (B, T, C) with C=80 and returns
    a normalized embedding of shape (B, embed_dim).
    """

    def __init__(
        self,
        channels: List[int],
        kernel_sizes: List[int],
        dilations: List[int],
        attention_channels: int,
        res2net_scale: int,
        se_channels: int,
        global_context: bool,
        groups: List[int],
        embed_dim: int,
        input_dim: int,
    ):
        super().__init__()
        assert len(channels) == len(kernel_sizes) == len(dilations)
        self.channels = channels
        self.embed_dim = embed_dim
        self.blocks = ModuleList()

        self.blocks.append(
            TDNNBlock(
                input_dim,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                groups[0],
            )
        )

        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    groups=groups[i],
                )
            )

        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            groups=groups[-1],
        )

        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_norm = LayerNorm(channels[-1] * 2, eps=1e-12)

        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
            out_channels=embed_dim,
            kernel_size=1,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        def encoder_init(m: Module) -> None:
            if isinstance(m, Conv1d):
                init.xavier_uniform_(m.weight, init.calculate_gain("relu"))

        self.apply(encoder_init)

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # x: (B, T, C)
        x = x.transpose(1, 2)  # (B, C, T)

        xl = []
        for layer in self.blocks:
            x = layer(x, padding_mask=padding_mask)
            xl.append(x)

        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        x = self.asp(x, padding_mask=padding_mask)
        x = self.asp_norm(x.transpose(1, 2)).transpose(1, 2)

        x = self.fc(x)

        x = x.transpose(1, 2).squeeze(1)  # (B, embed_dim)
        return F.normalize(x, dim=-1)


class TDNNBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
            groups=groups,
        )
        self.activation = ReLU()
        self.norm = LayerNorm(out_channels, eps=1e-12)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        x = self.activation(self.conv(x))
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class Res2NetBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int = 8,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        assert in_channels % scale == 0
        assert out_channels % scale == 0

        in_channel = in_channels // scale
        hidden_channel = out_channels // scale
        self.blocks = ModuleList(
            [
                TDNNBlock(
                    in_channel,
                    hidden_channel,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
                for _ in range(scale - 1)
            ]
        )
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        y = []
        for i, x_i in enumerate(torch.chunk(x, self.scale, dim=1)):
            if i == 0:
                y_i = x_i
            elif i == 1:
                y_i = self.blocks[i - 1](x_i)
            else:
                y_i = self.blocks[i - 1](x_i + y_i)
            y.append(y_i)
        return torch.cat(y, dim=1)


class SEBlock(Module):
    def __init__(
        self,
        in_channels: int,
        se_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.conv1 = Conv1d(in_channels=in_channels, out_channels=se_channels, kernel_size=1)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv1d(in_channels=se_channels, out_channels=out_channels, kernel_size=1)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        if padding_mask is not None:
            # padding_mask: (B, T) with 1 for valid, 0 for pad
            mask = padding_mask.unsqueeze(1)  # (B, 1, T)
            lengths = mask.sum(dim=2, keepdim=True)
            s = (x * mask).sum(dim=2, keepdim=True) / torch.clamp(lengths, min=1.0)
        else:
            s = x.mean(dim=2, keepdim=True)

        s = self.relu(self.conv1(s))
        s = self.sigmoid(self.conv2(s))
        return s * x


class AttentiveStatisticsPooling(Module):
    def __init__(
        self, channels: int, attention_channels: int = 128, global_context: bool = True
    ):
        super().__init__()
        self.eps = 1e-12
        self.global_context = global_context
        if global_context:
            self.tdnn = TDNNBlock(channels * 3, attention_channels, 1, 1)
        else:
            self.tdnn = TDNNBlock(channels, attention_channels, 1, 1)

        self.tanh = Tanh()
        self.conv = Conv1d(in_channels=attention_channels, out_channels=channels, kernel_size=1)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        # x: (N, C, L)
        N, C, L = x.shape

        def _compute_statistics(
            x: Tensor, m: Tensor, dim: int = 2, eps: float = 1e-12
        ) -> Tuple[Tensor, Tensor]:
            mean = (m * x).sum(dim)
            std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
            return mean, std

        if padding_mask is not None:
            mask = padding_mask
        else:
            mask = torch.ones(N, L, device=x.device, dtype=x.dtype)
        mask = mask.unsqueeze(1)  # (N, 1, L)

        if self.global_context:
            total = mask.sum(dim=2, keepdim=True).to(x)
            mean, std = _compute_statistics(x, mask / total)
            mean = mean.unsqueeze(2).repeat(1, 1, L)
            std = std.unsqueeze(2).repeat(1, 1, L)
            attn = torch.cat([x, mean, std], dim=1)
        else:
            attn = x

        attn = self.conv(self.tanh(self.tdnn(attn)))

        attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=2)
        mean, std = _compute_statistics(x, attn)
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(2)
        return pooled_stats


class SERes2NetBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res2net_scale: int = 8,
        se_channels: int = 128,
        kernel_size: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.tdnn1 = TDNNBlock(
            in_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            groups=groups,
        )
        self.res2net_block = Res2NetBlock(
            out_channels,
            out_channels,
            res2net_scale,
            kernel_size,
            dilation,
        )
        self.tdnn2 = TDNNBlock(
            out_channels,
            out_channels,
            kernel_size=1,
            dilation=1,
            groups=groups,
        )
        self.se_block = SEBlock(out_channels, se_channels, out_channels)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.tdnn1(x)
        x = self.res2net_block(x)
        x = self.tdnn2(x)
        x = self.se_block(x, padding_mask=padding_mask)
        return x + residual


def extract_fbank_16k(audio_16k: Tensor) -> Tensor:
    """
    Compute 80-dim FBANK features from 16kHz audio.

    Args:
        audio_16k: Tensor of shape (T,) or (1, T)
    Returns:
        fbank: Tensor of shape (T_fbank, 80)
    """
    if audio_16k.ndim == 1:
        audio_16k = audio_16k.unsqueeze(0)

    # Ensure minimum length for kaldi.fbank window (default 25ms @16k -> 400 samples)
    min_len = 400
    
    if audio_16k.shape[-1] < min_len:
        repeat_times = (min_len // audio_16k.shape[-1]) + 1
        audio_16k = audio_16k.repeat(1, repeat_times) if audio_16k.dim() > 1 else audio_16k.repeat(repeat_times)

    fbank = torchaudio.compliance.kaldi.fbank(
        audio_16k,
        num_mel_bins=80,
        sample_frequency=AUDIO_SAMPLE_RATE,
    )
    return fbank


class ProsodyEncoder(nn.Module):
    """
    High-level wrapper for the Pretssel prosody encoder.

    Usage:
        encoder = ProsodyEncoder(cfg_path, ckpt_path, freeze=True)
        emb = encoder(fbank_batch)  # (B, 512)
    """

    def __init__(self, cfg_path: Path, ckpt_path: Path, freeze: bool = True):
        super().__init__()
        model_cfg = self._load_pretssel_model_cfg(cfg_path)
        self.encoder = self._build_prosody_encoder(model_cfg)
        self._load_prosody_encoder_state(self.encoder, ckpt_path)
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    @staticmethod
    def _load_pretssel_model_cfg(cfg_path: Path) -> dict:
        cfg = json.loads(cfg_path.read_text())
        if "model" not in cfg:
            raise ValueError(f"{cfg_path} does not contain a top-level 'model' key.")
        return cfg["model"]

    @staticmethod
    def _build_prosody_encoder(model_cfg: dict) -> ECAPA_TDNN:
        encoder = ECAPA_TDNN(
            channels=model_cfg["prosody_channels"],
            kernel_sizes=model_cfg["prosody_kernel_sizes"],
            dilations=model_cfg["prosody_dilations"],
            attention_channels=model_cfg["prosody_attention_channels"],
            res2net_scale=model_cfg["prosody_res2net_scale"],
            se_channels=model_cfg["prosody_se_channels"],
            global_context=model_cfg["prosody_global_context"],
            groups=model_cfg["prosody_groups"],
            embed_dim=model_cfg["prosody_embed_dim"],
            input_dim=model_cfg["input_feat_per_channel"],
        )
        return encoder

    @staticmethod
    def _load_prosody_encoder_state(model: Module, ckpt_path: Path) -> None:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict):
            if all(isinstance(k, str) for k in state.keys()) and (
                any(k.startswith("prosody_encoder.") for k in state.keys())
                or any(k.startswith("prosody_encoder_model.") for k in state.keys())
            ):
                state = {
                    k.replace("prosody_encoder_model.", "", 1).replace("prosody_encoder.", "", 1): v
                    for k, v in state.items()
                    if k.startswith("prosody_encoder.") or k.startswith("prosody_encoder_model.")
                }
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Error loading checkpoint {ckpt_path}: missing keys={missing}, "
                f"unexpected keys={unexpected}"
            )

    def forward(self, fbank: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            fbank: Tensor of shape (B, T, 80)
            padding_mask: Optional tensor of shape (B, T) with 1 for valid.
        Returns:
            emb: Tensor of shape (B, 512)
        """
        return self.encoder(fbank, padding_mask=padding_mask)
