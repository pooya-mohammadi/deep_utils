from typing import List, Optional, Union, Tuple

import decord
import numpy as np


class DecordUtils:

    @staticmethod
    def get_fps(video_path: Optional[str] = None, vr: Optional[decord.VideoReader] = None) -> int:
        """
        Get video fps using decord
        :param video_path: path to video file
        :param vr: decord video reader
        :return:
        """

        assert video_path or vr, "Either video_path or vr must be provided"

        if vr is None:
            with open(video_path, mode="rb") as f:
                vr = decord.VideoReader(f)  # noqa
        output = round(vr.get_avg_fps())
        del vr
        return output

    @staticmethod
    def get_fps_frames(video_path: str, fps: int) -> List[int]:
        """
        Get frames with specific fps
        :param fps: fps of the video
        :param video_path: path to video file
        :return:
        """
        with open(video_path, mode="rb") as f:
            vr = decord.VideoReader(f)  # noqa
            org_fps = DecordUtils.get_fps(vr=vr)
            if fps > org_fps:
                raise ValueError(f"fps: {fps} must be less than or equal to original fps: {org_fps}")
            skip_size = org_fps / fps
            batch_frame = [int(i * skip_size) for i in range(int(len(vr) / skip_size))]
            del vr
        return batch_frame

    @staticmethod
    def read_video_indices(
            frame_indices: List[int],
            video_path: Optional[str] = None,
            vr: decord.VideoReader = None,
            width: int = None,
            height: int = None,
            fps: int = None,
            rgb: bool = True,
            return_vr: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, decord.VideoReader]]:
        """
        Get video indices using decord.
        :param video_path: path to video file
        :param frame_indices: a list of frame indices
        :param vr: Video Reader
        :param width: width of the video
        :param height: height of the video
        :param fps: fps of the video
        :param rgb: if True, returns the last 3 channels
        :param return_vr: return video_reader to be used in other sections!
        :return:
        """
        if fps is not None:
            fps_frames_indices = DecordUtils.get_fps_frames(video_path, fps)
            indices = list(np.array(fps_frames_indices)[frame_indices])
        else:
            indices = frame_indices

        if width is not None and height is not None:
            with open(video_path, mode="rb") as f:
                vr = decord.VideoReader(f, width=width, height=height)  # noqa
                indices = vr.get_batch(indices).asnumpy()
        else:
            if not vr:
                with open(video_path, mode="rb") as f:
                    vr = decord.VideoReader(f)  # noqa
            indices = vr.get_batch(indices).asnumpy()
        if rgb:
            indices = indices[..., :3]  # only the first 3 channels
        if return_vr:
            return indices, vr
        del vr
        return indices

    @staticmethod
    def count_video_frames(video_path: Optional[str], vr=None, return_vr: bool = False) -> Union[
        int, Tuple[int, decord.VideoReader]]:
        """
        Count video frames using decord
        :param video_path: path to video file
        :param vr: decord video reader
        :param return_vr: return video_reader to be used in other sections!
        :return:
        """
        if vr is None:
            with open(video_path, mode="rb") as f:
                vr = decord.VideoReader(f)  # noqa
        output = len(vr)
        if return_vr:
            return output, vr
        del vr
        return output
