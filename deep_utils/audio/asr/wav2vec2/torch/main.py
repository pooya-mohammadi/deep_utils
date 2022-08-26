from typing import List, Tuple
import librosa
import numpy as np
import torch
from torch.nn import Module
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


class Wav2Vec2STTTorch:
    def __init__(self, model_path, device="cpu", sample_rate=16000):
        self.device = device
        self.sample_rate = sample_rate
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model:Module = Wav2Vec2ForCTC.from_pretrained(
            model_path).eval().to(self.device)

    def stt(self, speech_array) -> str:
        if len(speech_array.shape) == 2:
            speech_array = speech_array.squeeze(0)
        with torch.no_grad():
            input_values = self.processor(
                speech_array, sampling_rate=self.sample_rate, return_tensors="pt"
            ).input_values.to(self.device)
            logits = self.model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            pred_str = self.processor.batch_decode(pred_ids)[0]
        return pred_str

    def stt_file(self, file_path) -> Tuple[np.ndarray, float, str]:
        speech, sr = librosa.load(file_path, self.sample_rate)
        return speech, sr, self.stt(speech)

    def stt_group(self, audio_segments) -> List[str]:
        texts = []
        for audio in audio_segments:
            text = self.stt(audio)
            texts.append(text)
        return texts
