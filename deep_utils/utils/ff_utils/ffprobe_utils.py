import subprocess


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


if __name__ == '__main__':
    fps = FFProbeUtils.get_fps("https://filmeditor.io/thumbnails/saeed-video/InformativeShortFormAnimation_DEAR.mp4")
    print("fps: ", fps)
