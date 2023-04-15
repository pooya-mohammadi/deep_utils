try:
    from deep_utils._dummy_objects.audio.vad.pyannote import PyannoteAudioVAD
    from .pyannote_audio_vad import PyannoteAudioVAD
except ModuleNotFoundError:
    pass
