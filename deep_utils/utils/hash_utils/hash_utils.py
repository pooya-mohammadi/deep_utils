def get_hash_file(file, buffer_size=65536):
    """
    This function converts a file in chunk mode to hash file!
    buffer_size = 65536 lets read stuff in 64kb chunks
    :param file: str, io.BufferedReader
    :param buffer_size:
    :return:
    """
    import hashlib
    hash_obj = hashlib.sha512()
    if isinstance(file, str):
        with open(file, 'rb') as f:
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                hash_obj.update(data)
    else:
        hash_obj = hashlib.sha512()
        while True:
            data = file.read(buffer_size)
            if not data:
                break
            hash_obj.update(data)
    return hash_obj.hexdigest()
