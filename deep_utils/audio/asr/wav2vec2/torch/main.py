from typing import Tuple
import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


class Wav2Vec2STTTorch:
    def __init__(self, model_path, device='cpu', sample_rate=16000):
        self.device = device
        self.sample_rate = sample_rate
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path).eval().to(self.device)

    def speech_to_text(self, speech_array) -> str:
        input_values = self.processor(speech_array, sampling_rate=self.sample_rate,
                                      return_tensors='pt').input_values.to(self.device)
        logits = self.model(input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = self.processor.batch_decode(pred_ids)[0]
        return pred_str

    def stt_file(self, file_path) -> Tuple[np.ndarray, float, str]:
        speech, sr = librosa.load(file_path, self.sample_rate)
        return speech, sr, self.speech_to_text(speech)
