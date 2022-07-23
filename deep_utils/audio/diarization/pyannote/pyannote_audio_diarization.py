from pyannote.audio import Pipeline
from deep_utils.utils.logging_utils.logging_utils import log_print, get_logger
from deep_utils.audio.audio_utils.torchaudio_utils import TorchAudioUtils
import torchaudio
# from pyannote.audio.pipelines import SpeakerSegmentation


class PyannoteAudioDiarization:
    def __init__(self, device="cpu", min_duration_on=1, max_duration_on=10, logger=None):
        self.max_duration_on = max_duration_on
        # self.diarize_model = SpeakerSegmentation("pyannote/segmentation")
        self.diarize_model = Pipeline.from_pretrained("pyannote/speaker-segmentation")
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

    def __call__(self, wave, sr=None, logger=None, verbose=1):
        model_input = dict({"waveform": wave, "sample_rate": sr})
        wave_note = f"wave-shape: {wave.shape}, sr: {sr}"
        out = self.diarize_model(model_input)
        indices, audio_segments = [], []
        for speech_region, _, speaker in out.itertracks(yield_label=True):
            w = wave[0][int(speech_region.start * sr):int(speech_region.end * sr)].unsqueeze(0)
            ws = TorchAudioUtils.split(w, sr=sr, max_seconds=self.max_duration_on, logger=logger, verbose=verbose)
            indices.extend([int(speaker.split('_')[-1]) for _ in range(len(ws))])
            audio_segments.extend(ws)
        log_print(logger, f"Successfully diarized wav-file: {wave_note} to {len(audio_segments)} segments")
        return indices, audio_segments

    def diarize_file(self, wave_path, logger=None):
        wave, sr = torchaudio.load(wave_path)
        indices, audio_segments = self(wave=wave, sr=sr, logger=logger)
        return indices, audio_segments

    def diarize_group(self, audio_segments, sr, logger=None, verbose=1):
        res_segments, suffix = [], []
        for i, audio_segment in enumerate(audio_segments):
            indices, segments = self(audio_segment, sr=sr, logger=logger)

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
    logger = get_logger("pyannote/diarize")
    audio_diarization = PyannoteAudioDiarization(logger=logger, device="cpu")
    audio_path = "/home/ai/projects/speech/dataset/irancel-voice-dataset/samples_01/all-data_segments/0809155/0809155_000_00_S00_F.wav"
    indices, audio_segments = audio_diarization.diarize_file(audio_path, logger)
    # audio_dirs = ["/home/ai/projects/speech/dataset/irancel-voice-dataset/activation_segments",
    #               "/home/ai/projects/speech/dataset/irancel-voice-dataset/old-audios/general_segments",
    #               "/home/ai/projects/speech/dataset/irancel-voice-dataset/old-audios/repairs_segments",
    #               "/home/ai/projects/speech/dataset/irancel-voice-dataset/old-audios/arrival_segments"]
    # for audio_dir in audio_dirs:
    #     for speech_name in os.listdir(audio_dir):
    #         speech_path = os.path.join(audio_dir, speech_name)
    #         for wave_name in os.listdir(speech_path):
    #             wave_path = os.path.join(speech_path, wave_name)
    #             indices, audio_segments = audio_diarization.diarize_file(wave_path)
    #             if indices:
    #                 for j, (index, segment) in enumerate(zip(indices, audio_segments)):
    #                     diarized_audio_path = split_extension(wave_path, suffix=f"_{j:02}_S{index:02}")
    #                     torchaudio.save(diarized_audio_path, segment, 16000)
    #                     log_print(logger, f"Successfully saved {segment.shape} to {diarized_audio_path}")
