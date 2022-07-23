from pyannote.audio import Pipeline
from deep_utils.utils.logging_utils.logging_utils import log_print
from deep_utils.audio.audio_utils.torchaudio_utils import TorchAudioUtils
from deep_utils.utils.dir_utils.dir_utils import split_extension
import torchaudio
from pyannote.audio.pipelines import VoiceActivityDetection


class PyannoteAudioVAD:
    def __init__(self, device="cpu", min_duration_on=1, max_duration_on=10, logger=None):
        self.max_duration_on = max_duration_on
        self.vad_pipeline = VoiceActivityDetection(segmentation="pyannote/segmentation", device=device)
        self.vad_pipeline.instantiate({
            # onset/offset activation thresholds
            "onset": 0.5, "offset": 0.5,
            # remove speech regions shorter than that many seconds.
            "min_duration_on": min_duration_on,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": 0
        })
        log_print(logger, "Successfully Created VAD!")
        self.speaker_segmentor_pipeline = Pipeline.from_pretrained("pyannote/speaker-segmentation")

    def __call__(self, wave, sr, logger=None, verbose=1):
        model_input = dict({"waveform": wave, "sample_rate": sr})
        speech_activities = self.vad_pipeline(model_input)
        audio_segments = []
        for i, speech_region in enumerate(speech_activities.get_timeline()):
            w = wave[0][int(speech_region.start * sr):int(speech_region.end * sr)].unsqueeze(0)
            ws = TorchAudioUtils.split(w, sr=sr, max_seconds=self.max_duration_on, logger=logger, verbose=verbose)
            audio_segments.extend(ws)
        log_print(logger, f"Successfully segmented wave-array to {len(audio_segments)} segments", verbose=verbose)
        return audio_segments

    def vad_save(self, wave_path, audio_dir=None, logger=None, verbose=1):
        wave, sr = torchaudio.load(wave_path)
        audio_segments = self(wave=wave, sr=sr, logger=logger, verbose=verbose)
        audio_dir = wave_path if audio_dir is None else audio_dir
        for index, segment in enumerate(audio_segments):
            audio_path = split_extension(audio_dir, suffix=f"_{index:03}")
            torchaudio.save(audio_path, segment, 16000)
        log_print(logger, f"Successfully saved wave to {audio_dir}", verbose=verbose)

# if __name__ == '__main__':
#     audio_vad = PyannoteAudioVAD()
#     audio_dirs = ["/home/ai/projects/speech/dataset/irancel-voice-dataset/old-audios/activation_segments",
#                   "/home/ai/projects/speech/dataset/irancel-voice-dataset/old-audios/general_segments",
#                   "/home/ai/projects/speech/dataset/irancel-voice-dataset/old-audios/repairs_segments",
#                   "/home/ai/projects/speech/dataset/irancel-voice-dataset/old-audios/arrival_segments"]
#     for audio_dir in audio_dirs:
#         for speech_name in os.listdir(audio_dir):
#             speech_path = os.path.join(audio_dir, speech_name)
#             for wave_name in os.listdir(speech_path):
#                 wave_path = os.path.join(speech_path, wave_name)
#                 audio_vad.vad_save(wave_path)
