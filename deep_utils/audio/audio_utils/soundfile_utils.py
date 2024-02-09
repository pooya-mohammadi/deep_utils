import soundfile as sf


class SoundFileUtils:
    @staticmethod
    def write(file_path: str, audio, sample_rate: int = 48000, sub_type: str = "PCM_24"):
        sf.write(file_path, audio, sample_rate, sub_type)

