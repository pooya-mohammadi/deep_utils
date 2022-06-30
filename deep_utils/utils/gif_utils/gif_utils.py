import numpy as np

from deep_utils.utils.logging_utils.logging_utils import log_print
import glob
from PIL import Image


def make_gif_dir(frame_folder, output_path, extension=".jpg", logger=None, verbose=1):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.{extension}")]
    make_gif(frames, output_path, logger=logger, verbose=verbose)


def make_gif(frames, output_path, logger=None, verbose=1):
    if isinstance(frames[0], np.ndarray):
        frames = [Image.fromarray(frame) for frame in frames]
    frame_one = frames[0]
    frame_one.save(output_path, format="GIF", append_images=frames,
                   save_all=True, duration=len(frames) // 10, loop=0)
    log_print(logger, f"Successfully saved gif to {output_path}", verbose=verbose)
