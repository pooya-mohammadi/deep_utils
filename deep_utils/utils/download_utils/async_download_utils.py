import os
os.environ['CURL_CA_BUNDLE'] = ''
import aiohttp
import asyncio
from os.path import join, split

class AsyncDownloadUtils:

    @staticmethod
    async def download(url: str, local_filepath: str =None, chunk_size: int = 69 * 1024, exists_ok=True, session = None, directory_path: str=None) -> str:
        """
        Download a file from a URL in an asynchronous manner into the input local filepath.
        If url is a local file, return url.
        :param url:
        :param local_filepath:
        :param chunk_size:
        :param exists_ok: if True, do not download if the file already exists.
        :return:
        """
        async def _write(local_filepath):
            response.raise_for_status()
            with open(local_filepath, 'wb') as file:
                while True:
                    chunk = await response.content.read(chunk_size)
                    if not chunk:
                        break
                    file.write(chunk)

        if os.path.exists(url):
            return url

        if local_filepath is None and directory_path is not None:
            os.makedirs(directory_path, exist_ok=True)
            local_filepath = join(directory_path, split(url)[-1])

        if local_filepath is not None and os.path.exists(local_filepath) and exists_ok:
            return local_filepath
        elif local_filepath is not None and os.path.exists(local_filepath) and not exists_ok:
            raise ValueError("file exists!")

        if session is None:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    await _write(local_filepath)
        else:
            async with session.get(url) as response:
                await _write(local_filepath)
        return local_filepath

    @staticmethod
    async def download_urls(urls: list[str], local_dir: str, remove_to_get_local_file_path: str = None,
                            async_tasks: int = None):
        os.makedirs(local_dir, exist_ok=True)
        tasks = []
        for url in urls:
            filename = url.replace(remove_to_get_local_file_path, "").strip("/")
            local_filepath = os.path.join(local_dir, filename)
            os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
            task = asyncio.create_task(AsyncDownloadUtils.download(url, local_filepath))
            tasks.append(task)
        if async_tasks is not None:
            local_files = []
            for i in range(0, len(tasks) // async_tasks):
                _tasks = tasks[i * async_tasks: (i + 1) * async_tasks]
                print(f"working on {len(_tasks)}")
                files = await asyncio.gather(*_tasks)
                local_files.extend(files)
        else:
            local_files = await asyncio.gather(*tasks)
        return local_files


if __name__ == '__main__':
    import asyncio

    dl = "http://188.245.157.94:9001/api/v1/download-shared-object/aHR0cDovLzEyNy4wLjAuMTo5MDAwL3ZpZGVvcy8xLzAwYWM5YTVhODI0NzQ5MjI4NjI4NGJjZTNhZmQ3YWUwLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUMyTFpQUlcyRERTWUhKMDY5S0lDJTJGMjAyNDEwMjYlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDI2VDE5NTkwN1omWC1BbXotRXhwaXJlcz00MzIwMCZYLUFtei1TZWN1cml0eS1Ub2tlbj1leUpoYkdjaU9pSklVelV4TWlJc0luUjVjQ0k2SWtwWFZDSjkuZXlKaFkyTmxjM05MWlhraU9pSkRNa3hhVUZKWE1rUkVVMWxJU2pBMk9VdEpReUlzSW1WNGNDSTZNVGN6TURBeE5UY3pNaXdpY0dGeVpXNTBJam9pYldsdWFXOGlmUS5Zbnc1LUlLekVuQzNvdjBMOUZUXzR0WEVNVjJWSmtrWWZlRTZQS2VUenVYdU02X3A3Y21LY0d3Zm1pWTlSVllNMGNacnV3REUwbDJudm16bGE3dHo1USZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmdmVyc2lvbklkPW51bGwmWC1BbXotU2lnbmF0dXJlPWMxMGJkNzA4ZmQ2Njc2YWQ5YTBjMzBjZDU5ZTAyOWQ5ZTU0M2M2ZmE4NTc0ZWE0YWFkNDE4NGJhOTc2YTIwMDM"

    asyncio.run(AsyncDownloadUtils.download(dl, "sample.mp4"))
