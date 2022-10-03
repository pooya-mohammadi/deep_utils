def download_binary(url: str):
    import urllib.request
    from io import BytesIO
    if not url.startswith('http'):
        url = "http://" + url
    response = urllib.request.urlopen(url)
    data = response.read()

    file = BytesIO(data)
    return file
