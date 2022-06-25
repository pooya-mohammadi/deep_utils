import os

from deep_utils.utils.logging_utils import log_print
from deep_utils.utils.os_utils.os_path import split_extension


def vox2wav(
    file_path,
    in_audio_type="v3",
    overwrite=True,
    wav_path: str = None,
    logger=None,
    verbose=1,
):
    if in_audio_type.lower() in ("vox", "v3"):
        wav_path = (
            split_extension(file_path, extension=".wav")
            if wav_path is None
            else wav_path
        )
        if overwrite:
            os.system(
                f'ffmpeg -f u8 -c adpcm_ima_oki -ar 6.0k -ac 1 -i "{file_path}" "{wav_path}" -y'
            )
        else:
            if os.path.isfile(wav_path):
                raise ValueError(
                    f"overwrite is set to {overwrite} while the file exists in {wav_path}"
                )
    else:
        wav_path = file_path
    log_print(
        logger, f"Successfully converted {file_path} to {wav_path}", verbose=verbose
    )
    return wav_path
