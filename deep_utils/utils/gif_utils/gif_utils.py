import numpy as np
from deep_utils.utils.logging_utils.logging_utils import log_print
import glob
from PIL import Image


class GIFUtils:
    @staticmethod
    def make_gif_dir(frame_folder, output_path, extension=".jpg", logger=None, verbose=1):
        frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.{extension}")]
        GIFUtils.make_gif(frames, output_path, logger=logger, verbose=verbose)

    @staticmethod
    def make_gif(frames, output_path, duration=None, logger=None, verbose=1):
        duration = duration if duration is not None else (len(frames) // 10)
        if isinstance(frames[0], np.ndarray):
            frames = [Image.fromarray(frame) for frame in frames]
        frame_one = frames[0]
        frame_one.save(output_path, format="GIF", append_images=frames,
                       save_all=True, duration=duration, loop=0)
        log_print(logger, f"Successfully saved gif to {output_path}", verbose=verbose)
