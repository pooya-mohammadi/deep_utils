def unzip(zip_path, zip_dir='.'):
    from zipfile import ZipFile
    with ZipFile(zip_path, 'r') as zip:
        print(f'extracting {zip_path}')
        zip.extractall(zip_dir)
        print(f'extracting is done!')
