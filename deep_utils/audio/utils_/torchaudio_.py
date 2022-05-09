from pathlib import Path
from typing import Union
import torchaudio
from torchaudio import transforms as T
import torch
from deep_utils.utils.os_utils.os_path import split_extension
from deep_utils.utils.utils.logging_ import log_print


class TorchAudioUtils:
    @staticmethod
    def resample(wave: Union[str, Path, torch.Tensor], sr: int = None, resample_rate=16000, save=False,
                 resampled_path: str = None, logger=None) -> Union[str, torch.Tensor]:

        if isinstance(wave, str) or isinstance(wave, Path):
            waveform, sample_rate = torchaudio.load(wave)
            if save:
                wave_path = split_extension(wave,
                                            suffix=f"_{resample_rate}") if resampled_path is None else resampled_path
        else:
            waveform = wave
            sample_rate = sr
            if save:
                wave_path = resampled_path

        resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
        resampled_waveform = resampler(waveform)
        if save:
            torchaudio.save(wave_path, resampled_waveform, resample_rate, encoding="PCM_S", bits_per_sample=16)
            log_print(logger,
                      f"Successfully resampled and saved wav-file to {wave_path} with {resample_rate} sample rate!")
            return wave_path
        else:
            log_print(logger, f"Successfully resampled wav-file {waveform.shape} with {resample_rate} sample rate!")
            return resampled_waveform
