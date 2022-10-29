from pyannote.audio import Pipeline
from deep_utils.utils.logging_utils.logging_utils import log_print, get_logger
from deep_utils.audio.audio_utils.torchaudio_utils import TorchAudioUtils
import torchaudio


class PyannoteAudioDiarization:
    def __init__(self, device="cpu", min_duration_on=1, max_duration_on=10, logger=None,
                 checkpoint_path="pyannote/speaker-segmentation", use_auth_token=None):
        self.max_duration_on = max_duration_on
        # self.diarize_model = SpeakerSegmentation("pyannote/segmentation")
        self.diarize_model = Pipeline.from_pretrained(checkpoint_path=checkpoint_path, use_auth_token=use_auth_token)
        self.diarize_model._segmentation.model.to(device)
        self.diarize_model._segmentation.device = device
        self.diarize_model.instantiate({
            # onset/offset activation thresholds
            "onset": 0.5, "offset": 0.5,
            # remove speech regions shorter than that many seconds.
            "min_duration_on": min_duration_on,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": 0
        })

        log_print(logger, "Successfully Created Speaker Diarization!")

    def infer(self, wave, sr, return_segments=True, logger=None, verbose=1):
        model_input = dict({"waveform": wave, "sample_rate": sr})
        wave_note = f"wave-shape: {wave.shape}, sr: {sr}"
        out = self.diarize_model(model_input)
        indices, audio_segments = [], []
        if return_segments:
            for speech_region, _, speaker in out.itertracks(yield_label=True):
                w = wave[0][int(speech_region.start * sr):int(speech_region.end * sr)].unsqueeze(0)
                ws = TorchAudioUtils.split(w, sr=sr, max_seconds=self.max_duration_on, logger=logger, verbose=verbose)
                indices.extend([int(speaker.split('_')[-1]) for _ in range(len(ws))])
                audio_segments.extend(ws)
            log_print(logger, f"Successfully diarized wav-file: {wave_note} to {len(audio_segments)} segments")
            return indices, audio_segments
        else:
            output = [(speaker, (speech_region.start, speech_region.end)) for speech_region, _, speaker in
                      out.itertracks(yield_label=True)]
            log_print(logger, f"Successfully diarized wav-file: {wave_note} to {len(audio_segments)} segments")
            return output

    def infer_file(self, wave_path, logger=None):
        wave, sr = torchaudio.load(wave_path)
        indices, audio_segments = self.infer(wave=wave, sr=sr, logger=logger)
        return indices, audio_segments

    def infer_group(self, audio_segments, sr, logger=None, verbose=1):
        res_segments, suffix = [], []
        for i, audio_segment in enumerate(audio_segments):
            indices, segments = self.infer(audio_segment, sr=sr, logger=logger)

            if indices:
                for j, (index, segment) in enumerate(zip(indices, segments)):
                    res_segments.append(segment)
                    suffix.append(f"{i:03}_{j:02}_S{index:02}")
            else:
                res_segments.append(audio_segment)
                suffix.append(f"{i:03}_{0:02}_S{0:02}")
        log_print(logger, f"Successfully diarized {len(suffix)} samples", verbose=verbose)
        return res_segments, suffix


if __name__ == '__main__':
    a = PyannoteAudioDiarization()
    a.infer(logger=1)
