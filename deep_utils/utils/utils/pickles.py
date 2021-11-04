def dump_pickle(file_path: str, file, mode: str = 'wb'):
    import pickle
    with open(file_path, mode=mode) as f:
        pickle.dump(file, f)


def load_pickle(file_path: str, mode: str = 'rb'):
    import pickle
    with open(file_path, mode=mode) as f:
        return pickle.load(f)
