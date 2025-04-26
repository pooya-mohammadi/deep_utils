import os
import shutil
import sys
from os.path import join, split
import requests


class DownloadUtils:
    @staticmethod
    def download_binary(url: str):
        import urllib.request
        from io import BytesIO
        if not url.startswith('http'):
            url = "http://" + url
        response = urllib.request.urlopen(url)
        data = response.read()

        file = BytesIO(data)
        return file

    @staticmethod
    def download_file(
            url,
            download_dir=".",
            filename=None,
            remove_download=False,
            exists_skip=False,
    ):
        """
        Download a file from url
        :param url:
        :param download_dir:
        :param filename:
        :param remove_download:
        :param exists_skip: If True, skip download if file exists
        :return:
        """
        if url is None:
            ValueError("url is None. Exiting the function")

        os.makedirs(download_dir, exist_ok=True)

        error_msg = "URL fetch failure on {}"
        temp_download_des = download_des = None

        try:
            response = requests.get(url, stream=True)
            total = response.headers.get("content-length")
            while not total:
                response = requests.get(url, stream=True)
                total = response.headers.get("content-length")
            try:
                if filename is None:
                    filename = response.headers.get("filename")
                    if filename is None:
                        filename = response.headers.get("content-disposition").split("=")[
                            -1
                        ]
            except:
                filename = os.path.split(url)[-1]

            download_des = os.path.join(download_dir, filename)
            temp_download_des = download_des + ".tmp"
            if exists_skip and os.path.isfile(download_des):
                return download_des
            with open(temp_download_des, "wb") as f:
                if total is None:
                    f.write(response.content)
                else:
                    downloaded = 0
                    total = int(total)
                    for data in response.iter_content(
                            chunk_size=max(int(total / 1000), 1024 * 1024)
                    ):
                        downloaded += len(data)
                        f.write(data)
                        done = int(50 * downloaded / total)
                        sys.stdout.write("\rDownloading {}: {}% [{}{}]"
                                         .format(filename, round((downloaded * 100 / total), 2), "█" * done,
                                                 "." * (50 - done)))
                        sys.stdout.flush()
            sys.stdout.write("\n")
            shutil.move(temp_download_des, download_des)
        except (Exception, KeyboardInterrupt):
            if (
                    temp_download_des is not None
                    and os.path.isfile(temp_download_des)
                    and remove_download
            ):
                os.remove(temp_download_des)
            raise Exception(error_msg.format(url))
        return download_des

    def download_urls(dl_urls: list[str], download_path: str, remove_to_get_local_file_path: str = None):
        for url in dl_urls:
            if remove_to_get_local_file_path:
                filename = url.replace(remove_to_get_local_file_path, "").strip("/")
            local_filepath = os.path.join(download_path, filename)
            os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
            DownloadUtils.download_file(url, os.path.dirname(local_filepath), filename=split(filename)[-1].replace('"', "").replace(";", ""), exists_skip=True)

if __name__ == '__main__':
    image_download_path = "https://github.com/pooya-mohammadi/deep_utils/releases/download/1.0.2/golsa_in_garden.jpg"
    DownloadUtils.download_file(image_download_path, exists_skip=False)
