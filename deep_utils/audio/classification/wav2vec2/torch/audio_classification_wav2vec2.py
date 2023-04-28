from os.path import join
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from typing import Union
from pathlib import Path
import torch
import torchaudio
import librosa
import numpy as np
from deep_utils.utils.pickle_utils.pickle_utils import load_pickle
from deep_utils.utils.logging_utils.logging_utils import log_print
from deep_utils.audio.audio_utils.torchaudio_utils import TorchAudioUtils


class AudioClassificationWav2vec2Torch:
    def __init__(self, model_path, num_labels, device="cpu", sr=16_000, label2id="label2id.pkl",
                 feature_extractor="facebook/wav2vec2-base", logger=None, verbose=1):
        """

        :param num_labels:
        :param model_path: A path to a directory which contains pytorch_model.bin, training_args.bin, and label2id variable
        :param feature_extractor:
        """
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor)
        self.label2id = load_pickle(join(model_path, label2id))
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.sr = sr
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_path, num_labels=num_labels, label2id=self.label2id, id2label=self.id2label
        ).to(device)
        self.model = self.model.eval()
        self.device = device
        self.logger = logger
        self.verbose = verbose
        log_print(self.logger, "Successfully created Wav2Vec2 Classification Object")

    def infer(self, wave: Union[Path, str, np.ndarray, torch.Tensor], sr=None):
        if isinstance(wave, (Path, str)):
            waveform, sr = librosa.load(wave)
            waveform = torch.from_numpy(waveform).unsqueeze(0)
        else:
            waveform = wave
        waveform = TorchAudioUtils.resample(waveform, sr, self.sr)
        # waveform = torchaudio.transforms.Resample(sr, self.sr)(waveform)
        inputs = self.feature_extractor(waveform, sampling_rate=self.feature_extractor.sampling_rate,
                                        max_length=16000, truncation=True)
        tensor = torch.tensor(inputs['input_values'][0]).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            logits = output['logits'][0]
            label_id = torch.argmax(logits).item()
        label_name = self.id2label[str(label_id)]
        return label_name

    def infer_group(self, audio_segments, sr, indices=None):
        suffix = []
        indices = ["" for _ in range(len(audio_segments))] if indices is None else indices
        delimiter = '_' if indices is not None else ''
        for index, audio_segment in zip(indices, audio_segments):
            gender = self.infer(audio_segment, sr=sr)
            suffix.append(str(index) + f"{delimiter}{gender}")
        log_print(self.logger, f"Successfully performed {len(suffix)} gender detection", verbose=self.verbose)
        return suffix
