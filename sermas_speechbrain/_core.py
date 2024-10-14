from typing import Self
import dataclasses

import torch
import torchaudio


@dataclasses.dataclass
class Audio:

    waveform: torch.Tensor
    sample_rate: int

    @classmethod
    def from_file(cls, audiofile) -> Self:
        waveform, sample_rate = torchaudio.load(audiofile)
        return cls(waveform=waveform, sample_rate=sample_rate)
