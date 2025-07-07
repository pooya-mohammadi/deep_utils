import os
import subprocess

from deep_utils.utils.dir_utils.dir_utils import DirUtils


class DownloadYoutube:
    """
    Requirements:
    requests>=2.28.1
    beautifulsoup4>=4.11.1
    pytube>=12.1.0
    tqdm>=4.64.1
    yt-dlp>=2023.3.4
    """

    @staticmethod
    def get_format_string(file_format="mp4", resolution="best") -> str:
        """
        Get the yt-dlp format string based on configuration.

        Args:
            file_format: Downloader configuration
            resolution: Downloader configuration

        Returns:
            Format string for yt-dlp
        """
        if resolution == "best":
            return f"bestvideo[ext={file_format}]+bestaudio/best[ext={file_format}]/best"
        elif resolution == "worst":
            return f"worstvideo[ext={file_format}]+worstaudio/worst[ext={file_format}]/worst"
        else:
            # Try to get specific resolution
            return f"bestvideo[height<={resolution}][ext={file_format}]+bestaudio/best[height<={resolution}][ext={file_format}]/best[height<={resolution}]"

    @staticmethod
    def download_with_ytdlp_cli(video_id: str, output_path: str, video_title: str, file_format="mp4",
                                resolution="best") -> bool:
        """
        Download a video using the yt-dlp command line.

        Args:
            video_id: Dictionary containing video information
            video_title: Dictionary containing video information
            output_path: Full path where to save the video

        Returns:
            True if download successful, False otherwise
        """
        if video_id.startswith("https://"):
            video_url = video_id
        else:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
        # video_title = video_title or video_id
        format_string = DownloadYoutube.get_format_string(file_format, resolution)

        # Build the yt-dlp command
        cmd = [
            'yt-dlp',
            '-f', format_string,
            '-o', output_path,
            '--no-warnings',
            '--progress',
            video_url
        ]

        try:
            print(f"Running yt-dlp command: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            # output = os.system(f"yt-dlp -f {format_string} -o {output_path} --no-warnings --progress {video_url}")
            stdout, stderr = process.communicate()
            print(f"Downloading {video_title}")
            if stderr != 0:
                print(f"yt-dlp error: {stderr}")
                return False

            print(f"Successfully downloaded video using yt-dlp command line")

            # Try to extract format information from stdout/stderr
            resolution_match = None
            # for line in stdout.split('\n') + stderr.split('\n'):
            #     if 'x' in line and ('p' in line or 'resolution' in line.lower()):
            #         # Try to find something like "1280x720" or "720p"
            #         import re
            #         res_pattern = r'(\d+x\d+|(\d+)p)'
            #         matches = re.findall(res_pattern, line)
            #         if matches:
            #             resolution_match = matches[0][0]
            #             break

            # Update video_info with the format that was downloaded
            # downloaded_format = {
            #     'format_id': 'yt-dlp_cli',
            #     'resolution': resolution_match or 'unknown',
            #     'downloaded': True
            # }
            # video_info['downloaded_format'] = downloaded_format

            return True
        except Exception as e:
            print(f"Error running yt-dlp command for {video_id}: {str(e)}")
            return False

    @staticmethod
    def download_video(video_id: str, output_dir: str, channel_name: str = None,
                       video_title: str = None, file_format="mp4", resolution="best") -> bool:
        """
        Download a single YouTube video.

        Args:
            video_info: Dictionary containing video information
            config: Downloader configuration

        Returns:
            True if download successful, False otherwise
        """
        if not video_id:
            print("[ERROR] Video ID missing from video info")
            return False
        video_title = video_title or video_id
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Create a subdirectory for this channel
        channel_name = channel_name or 'unknown_channel'
        channel_dir = DirUtils.sanitize_filename(channel_name)
        channel_path = os.path.join(output_dir, channel_dir)
        os.makedirs(channel_path, exist_ok=True)

        # Prepare filename
        filename = f"{DirUtils.sanitize_filename(video_title)}.{file_format}"
        output_path = os.path.join(channel_path, filename)

        # Check if file already exists
        if os.path.exists(output_path):
            print(f"[INFO] Video already downloaded: {output_path}")
            return True

        # Try to download with yt-dlp
        try:
            print(f"[INFO] Initializing download for video ID: {video_id}")

            success = DownloadYoutube.download_with_ytdlp_cli(video_id, output_path, video_title, file_format,
                                                              resolution)

            if success:
                # Create metadata file
                # metadata_path = os.path.join(channel_path, f"{DirUtils.sanitize_filename(video_title)}_{video_id}_info.json")
                # JsonUtils.dump(metadata_path, video_info)
                # with open(metadata_path, 'w', encoding='utf-8') as f:
                #     json.dump(video_info, f, ensure_ascii=False, indent=2)

                # print(f"[INFO] Successfully downloaded: {video_title}")
                return True
            else:
                print(f"[ERROR] Failed to download video {video_id}")
                return False

        except Exception as e:
            print(f"[ERROR] Error in download process for video {video_id}: {str(e)}")
            # Print traceback for debugging
            import traceback
            print(f"[ERROR] {traceback.format_exc()}")
            return False


if __name__ == '__main__':
    DownloadYoutube.download_video(video_id="jEH1eokufjU&list=PLYZxc42QNctWxkUZ7WSsUYC7yo8D85igl",
                                   output_dir="/home/aicvi/Downloads",
                                   channel_name="yasir_qadhi",
                                   video_title="Akhlagh_01",
                                   resolution="worst")
