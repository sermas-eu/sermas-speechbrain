from typing import Self
import dataclasses

import soundfile
import torch


@dataclasses.dataclass
class Audio:

    waveform: torch.Tensor  # (channels, frames), float32 in [-1, 1]
    sample_rate: int

    @classmethod
    def from_file(cls, audiofile) -> Self:
        # soundfile (libsndfile): WAV and the other formats it knows, no FFmpeg needed
        data, sample_rate = soundfile.read(audiofile, dtype="float32", always_2d=True)
        return cls(waveform=torch.from_numpy(data.T.copy()), sample_rate=sample_rate)

    def to_dict(self):
        return dataclasses.asdict(self)
