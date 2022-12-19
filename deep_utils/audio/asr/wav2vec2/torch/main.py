from typing import List, Tuple, Union
import librosa
import numpy as np
import torch
from torch.nn import Module
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from deep_utils.audio.audio_utils.torchaudio_utils import TorchAudioUtils


class Wav2Vec2STTTorch:
    def __init__(self, model_path, device="cpu", sample_rate=16000, ctcdecoder=False, kenlm_model_path=None):
        self.device = device
        self.sample_rate = sample_rate
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model: Module = Wav2Vec2ForCTC.from_pretrained(
            model_path).eval().to(self.device)
        self.ctcdecoder = ctcdecoder
        if ctcdecoder:
            from pyctcdecode import build_ctcdecoder
            self.decoder = build_ctcdecoder(list(self.processor.tokenizer.get_vocab().keys()),
                                            kenlm_model_path=kenlm_model_path)

    def stt(self, speech_array, sr, mean=True) -> Union[str, torch.Tensor]:
        """

        :param speech_array:
        :param sr:
        :param mean: if set to True, get the mean of channels that are more than 1
        :return:
        """
        if len(speech_array.shape) == 2 and speech_array.shape[0] != 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)
        if len(speech_array.shape) == 2:
            speech_array = speech_array.squeeze(0)
        if sr != self.sample_rate:
            speech_array = TorchAudioUtils.resample(speech_array, sr, self.sample_rate)
        with torch.no_grad():
            input_values = self.processor(
                speech_array, sampling_rate=self.sample_rate, return_tensors="pt"
            ).input_values.to(self.device)
            logits = self.model(input_values).logits
            if self.ctcdecoder:
                logits = logits.cpu().numpy()[0]
                return self.decoder.decode(logits)
            else:
                pred_ids = torch.argmax(logits, dim=-1)
                pred_str = self.processor.batch_decode(pred_ids)[0]
        return pred_str

    def stt_file(self, file_path) -> Tuple[np.ndarray, float, str]:
        speech, sr = librosa.load(file_path, self.sample_rate)
        return speech, sr, self.stt(speech, sr)

    def stt_group(self, audio_segments, sr) -> List[str]:
        texts = []
        for audio in audio_segments:
            text = self.stt(audio, sr)
            texts.append(text)
        return texts
