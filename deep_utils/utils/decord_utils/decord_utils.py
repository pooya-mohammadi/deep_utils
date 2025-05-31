from typing import List, Optional, Union, Tuple, Any

import decord
import numpy as np

from deep_utils.utils.datetime_utils.datetime_utils import DateTimeUtils


class DecordUtils:

    @staticmethod
    def get_fps(video_path: Optional[str] = None, vr: Optional[decord.VideoReader] = None,
                round_output: bool = True, return_vr: bool=False) -> Union[int | float | Tuple[int | float, Any]]:
        """
        Get video fps using decord
        :param video_path: path to video file
        :param vr: decord video reader
        :param round_output:
        :param return_vr:
        :return:
        """

        assert video_path or vr, "Either video_path or vr must be provided"

        if vr is None:
            with open(video_path, mode="rb") as f:
                vr = decord.VideoReader(f)  # noqa
        output = vr.get_avg_fps()
        if round_output:
            output = round(output)
        if return_vr:
            return output, vr
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
                indices = vr.get_batch(indices)
                if isinstance(indices, decord.ndarray.NDArray):
                    indices = indices.asnumpy()
                else:
                    indices = indices.numpy()
        else:
            if not vr:
                with open(video_path, mode="rb") as f:
                    vr = decord.VideoReader(f)  # noqa
            indices = vr.get_batch(indices)
            if isinstance(indices, decord.ndarray.NDArray):
                indices = indices.asnumpy()
            else:
                indices = indices.numpy()
        if rgb:
            indices = indices[..., :3]  # only the first 3 channels
        if return_vr:
            return indices, vr
        del vr
        return indices

    @staticmethod
    def count_video_frames(video_path: str = None, vr=None, return_vr: bool = False) -> Union[
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

    @staticmethod
    def crop_video(video_path: str, start: str | float = "00:00:00", end: str | float= None, output_video_path: str = None, reverse: bool = False):
        """

        :param video_path:
        :param output_video_path:
        :param start: "01:09:23"
        :param end: "01:56:00"
        :param reverse:
        :return:
        """
        from deep_utils.utils.opencv_utils.opencv_utils import VideoWriterCV
        from deep_utils.utils.ff_utils.ffprobe_utils import FFProbeUtils
        if end is None:
            end = DecordUtils.get_video_duration(video_path)

        if output_video_path is None:
            raise ValueError("output_video_path cannot be None!")

        with open(video_path, mode="rb") as f:
            vr = decord.VideoReader(f)  # noqa

        if isinstance(end, (float, int)):
            end = DateTimeUtils.parse_str_time(end)
        if isinstance(start, (float, int)):
            start = DateTimeUtils.parse_str_time(start)
        num_frames = DecordUtils.count_video_frames(video_path=None, vr=vr)

        width, height, fps = FFProbeUtils.get_width_height_fps(video_path)
        start_frame = round(DateTimeUtils.parse_time_str(start) * fps)
        end_frame = round(DateTimeUtils.parse_time_str(end) * fps)
        frame_indices = []
        for item in list(range(start_frame, end_frame)):
            if item >= num_frames:
                print(
                    f"[WARNING] timing is not correct. Requesting for frame: {item} while available frames are {num_frames}")
                continue
            frame_indices.append(item)
        frames = DecordUtils.read_video_indices(frame_indices, vr=vr)
        vw = VideoWriterCV(output_video_path, height, width, 'mp4v', fps=fps, colorful=True)

        if reverse:
            frames = frames[::-1]

        for frame in frames:
            vw.write(frame[..., ::-1])

        vw.release()

    @staticmethod
    def get_all_frames(video_path: str, return_vr: bool=False) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        n_frames, vr = DecordUtils.count_video_frames(video_path, return_vr=True)
        frames = DecordUtils.read_video_indices(list(range(n_frames)), vr=vr)
        if return_vr:
            return frames, vr
        return frames

    @staticmethod
    def get_video_duration(video_path: str, output_in_string: bool = False):
        fps = DecordUtils.get_fps(video_path)
        all_frames = DecordUtils.count_video_frames(video_path)
        seconds = all_frames/fps
        if output_in_string:
            return DateTimeUtils.parse_str_time(seconds)
        else:
            return seconds


if __name__ == '__main__':
    # fps = DecordUtils.get_fps("https://filmeditor.io/thumbnails/saeed-video/InformativeShortFormAnimation_DEAR.mp4")
    # print("fps: ", fps)
    # s_time, e_time = "00:18:45", "00:18:55"
    # DecordUtils.crop_video("/home/aicvi/Downloads/Emily_in_Paris_S01E02_10bit_x265_1080p_WEB-DL_30nama_30NAMA.mkv",
    #                        s_time, e_time,
    #                        "demo.mp4")
    vid_path = "/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/pooya/cloth/vivid-sister/emily/S01E09_data/Emily_in_Paris_S01E09_10bit_x265_1080p_WEB-DL_30nama_30NAMA-Scene-025/agnostic/vid_00.mp4"
    vid_duration = DecordUtils.get_video_duration(vid_path)
    half = vid_duration/2
    output = DecordUtils.crop_video(vid_path, end=half, reverse=True, output_video_path="here.mp4")
    # print(output)
