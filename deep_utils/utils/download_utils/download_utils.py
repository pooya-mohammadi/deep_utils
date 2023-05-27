import sys
import requests
import os


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
            file_name=None,
            remove_download=False,
    ):
        """
        Download a file from url
        :param url:
        :param download_dir:
        :param file_name:
        :param remove_download:
        :return:
        """
        if url is None:
            ValueError("url is None. Exiting the function")

        os.makedirs(download_dir, exist_ok=True)

        error_msg = "URL fetch failure on {}"
        download_des = None
        try:
            response = requests.get(url, stream=True)
            total = response.headers.get("content-length")
            while not total:
                response = requests.get(url, stream=True)
                total = response.headers.get("content-length")
            try:
                if file_name is None:
                    file_name = response.headers.get("filename")
                    if file_name is None:
                        file_name = response.headers.get("content-disposition").split("=")[
                            -1
                        ]
            except:
                file_name = os.path.split(url)[-1]

            download_des = os.path.join(download_dir, file_name)
            with open(download_des, "wb") as f:
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
                        sys.stdout.write("\rDownloading: [{}{}]".format("█" * done, "." * (50 - done)))
                        sys.stdout.flush()
            sys.stdout.write("\n")
        except (Exception, KeyboardInterrupt):
            if (
                    download_des is not None
                    and os.path.isfile(download_des)
                    and remove_download
            ):
                os.remove(download_des)
            raise Exception(error_msg.format(url))
        return download_des



