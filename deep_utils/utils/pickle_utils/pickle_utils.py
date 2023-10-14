import pickle


class PickleUtils:

    @staticmethod
    def dump_pickle(file_path: str, file, mode: str = "wb"):
        with open(file_path, mode=mode) as f:
            pickle.dump(file, f)

    @staticmethod
    def load_pickle(file_path: str, mode: str = "rb", encoding=""):
        with open(file_path, mode=mode) as f:
            return pickle.load(f, encoding=encoding)
