# cp from https://github.com/lifeiteng/vall-e/blob/main/valle/data/tokenizer.py
# Copyright    2023                            (authors: Feiteng Li)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re, logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Pattern, Union
import math
import numpy as np
import torch
import torchaudio
# from lhotse.features import FeatureExtractor
# from lhotse.utils import Seconds, compute_num_frames
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

# Configure espeak-ng via espeakng_loader if available.
# This provides a consistent libespeak-ng + data across environments (e.g. HF Spaces).
try:
    import espeakng_loader

    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    data_path = espeakng_loader.get_data_path()
    # Export data path via environment so underlying espeak-ng uses it.
    os.environ["ESPEAK_DATA_PATH"] = data_path
    os.environ["ESPEAKNG_DATA_PATH"] = data_path
    print("[LEMAS-TTS] espeak-ng configured via espeakng_loader")
except Exception as e:  # ImportError or runtime errors
    # Fall back to system espeak-ng discovery.
    print(f"[LEMAS-TTS] espeakng_loader not available or failed ({e}); using system espeak-ng")


class TextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        backend="espeak",
        separator=Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        phonemizer = EspeakBackend(
            language,
            punctuation_marks=punctuation_marks,
            preserve_punctuation=preserve_punctuation,
            with_stress=with_stress,
            tie=tie,
            language_switch=language_switch,
            words_mismatch=words_mismatch,
        )
        
        self.backend = phonemizer
        self.separator = separator

    def to_list(self, phonemized: str) -> List[str]:
        fields = []
        for word in phonemized.split(self.separator.word):
            # "ɐ    m|iː|n?"    ɹ|ɪ|z|ɜː|v; h|ɪ|z.
            pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
            fields.extend(
                [p for p in pp if p != self.separator.phone]
                + [self.separator.word]
            )
        assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
            self.separator.phone
        )
        return fields[:-1]

    def __call__(self, text, strip=True) -> List[List[str]]:
        if isinstance(text, str):
            text = [text]
        phones = []
        for txt in text:
            if txt == '':
                continue
            if txt[0] == '#':
                phones.append(txt)
            else:
                ipa = text_tokenizer.backend.phonemize([txt], separator=text_tokenizer.separator, strip=True, njobs=1, logger=logging.basicConfig(level=logging.ERROR))
                phones += text_tokenizer.to_list(ipa[0])
        return phones


def tokenize_text(tokenizer: TextTokenizer, text: str) -> List[str]:
    phonemes = tokenizer([text.strip()])
    return phonemes[0]  # k2symbols


_PAUSE_SYMBOL = {'、':',', '，':',', '。':',', '！':'!', '？':'?', '：':':'}
def _replace(match):
    word = match.group(0)
    return _PAUSE_SYMBOL[word]

def txt2phone(tokenizer: TextTokenizer, text: str):
    text = re.sub('|'.join(_PAUSE_SYMBOL.keys()), _replace, text)
    text = re.split(r"(#\d)", text)
    phones = []
    for txt in text:
        if txt == '':
            continue
        if txt[0] == '#':
            phones.append(txt)
        else:
            ipa = tokenizer.backend.phonemize([txt], separator=tokenizer.separator, strip=True, njobs=1)
            phones += tokenizer.to_list(ipa[0])
    phones = "|".join(phones).replace("(|", "(").replace("|)", ")")
    # phones = ["(cmn)"] + phones.split("|")
    return phones   


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


class AudioTokenizer:
    """EnCodec audio."""

    def __init__(
        self,
        device: Any = None,
        signature = None
    ) -> None:
        from audiocraft.solvers import CompressionSolver
        model = CompressionSolver.model_from_checkpoint(signature)
        self.sample_rate = model.sample_rate
        self.channels = model.channels
        
        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")

        self._device = device

        self.codec = model.to(device)

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        codes = self.codec.encode(wav.to(self.device))
        return [(codes[0], None)]

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames[0][0] # [1,4,T]
        return self.codec.decode(frames)
    


def tokenize_audio(tokenizer: AudioTokenizer, audio, offset = -1, num_frames=-1):
    # Load and pre-process the audio waveform
    if type(audio) == str:
        if offset != -1 and num_frames!=-1:
            wav, sr = torchaudio.load(audio, frame_offset=offset, num_frames=num_frames)
        else:
            wav, sr = torchaudio.load(audio)
        wav = convert_audio(wav, sr, tokenizer.sample_rate, tokenizer.channels)
        wav = wav.unsqueeze(0)
    else:
        wav = audio.unsqueeze(0).unsqueeze(0)
    # Extract discrete codes from EnCodec
    with torch.no_grad():
        encoded_frames = tokenizer.encode(wav)
    return encoded_frames


class AudioSR:
    """EnCodec audio."""

    def __init__(
        self,
        model_path,
        device = "cpu",
    ) -> None:
        import dac
        self.codec = dac.DAC.load(model_path)
        self.codec.to(device)
        self.codec.eval()

        self.sample_rate = self.codec.sample_rate
        self.channels = 1
        self._device = device

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        length = wav.shape[-1]
        right_pad = math.ceil(length / self.codec.hop_length) * self.codec.hop_length - length
        wav = torch.nn.functional.pad(wav, (0, right_pad))
        z, codes, _, _, _ = self.codec.encode(wav.to(self._device))
        return [(codes, z)]

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        # frames = frames[0][0] # [1,4,T]
        # with torch.no_grad():
        #     z = self.codec.quantizer.from_codes(frames)[0]
        #     y = self.codec.decode(z)
        z = frames[0][1] # [1, 2048, T]
        with torch.no_grad():
            y = self.codec.decode(z)
        return y
