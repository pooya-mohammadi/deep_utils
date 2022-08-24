try:
    from deep_utils.dummy_objects.audio.utils_ import TorchAudioUtils
    from deep_utils.audio.audio_utils.torchaudio_utils import TorchAudioUtils
except ModuleNotFoundError:
    pass

try:
    from deep_utils.dummy_objects.audio.utils_ import LibrosaUtils
    from deep_utils.audio.audio_utils.librosa_utils import LibrosaUtils
except ModuleNotFoundError:
    pass

from .vox_utils import vox2wav
