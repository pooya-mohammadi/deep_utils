import subprocess
from typing import Tuple, Union



class FFProbeUtils:

    @staticmethod
    def get_fps(video_path: str) -> int:
        """
        Extracts the frame rate of a video
        :param video_path: path to video
        :return: frame rate
        """
        command = f"ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate {video_path}"
        frame_rate = subprocess.run(command, shell=True, capture_output=True, text=True)
        fps_vid = eval(frame_rate.stdout)
        return round(fps_vid)


    @staticmethod
    def get_video_size_url(dl_url) -> Tuple[int, int]:

        command = [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=width,height',
            '-of',
            'csv=p=0',
            dl_url
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            footage_size = result.stdout.strip().strip(",")
            width, height = [round(item.strip()) for item in footage_size.split(",")][:2]
            return width, height
        else:
            raise ValueError(f"dl_url: {dl_url} not working")

    @staticmethod
    def get_width_height_fps(dl_url:str) -> Tuple[int, int, int]:

        command = [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=width,height,r_frame_rate',
            '-of',
            'csv=p=0',
            dl_url
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            footage_size = result.stdout.strip().strip(",")
            width, height, fps = [round(float(item.strip().split("/")[0])) for item in footage_size.split(",")][:3]
            return width, height, fps
        else:
            raise ValueError(f"dl_url: {dl_url} not working")

    @staticmethod
    def get_width_height_fps_duration_frame_count(dl_url: str) -> Tuple[int, int, int, float, int]:

        command = [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=width,height,r_frame_rate,duration',
            '-of',
            'csv=p=0',
            dl_url
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            footage_size = result.stdout.strip().strip(",")
            width, height, fps, duration = [round(float(item.strip().split("/")[0])) if index < 3 else float(item.strip().split("/")[0]) for index, item in enumerate(footage_size.split(",")) ][:4]
            return width, height, fps, duration, round(duration * fps)
        else:
            raise ValueError(f"dl_url: {dl_url} not working")

if __name__ == '__main__':
    info = FFProbeUtils.get_width_height_fps_duration_frame_count("/home/aicvi/projects/ShifterRemotion/public/sample.mp4")
    print("info: ", info)
