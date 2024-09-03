import pickle

from deep_utils.utils.decorators.main import method_deprecated


class PickleUtils:

    @staticmethod
    def dump(file_path: str, file, mode: str = "wb"):
        with open(file_path, mode=mode) as f:
            pickle.dump(file, f)

    @staticmethod
    def load(file_path: str, mode: str = "rb", encoding=""):
        with open(file_path, mode=mode) as f:
            return pickle.load(f, encoding=encoding)

    @staticmethod
    @method_deprecated
    def load_pickle(file_path: str, mode: str = "rb", encoding=""):
        return PickleUtils.load(file_path=file_path, mode=mode, encoding=encoding)

    @staticmethod
    @method_deprecated
    def dump_pickle(file_path: str, file, mode: str = "wb"):
        return PickleUtils.dump(file_path=file_path, file=file, mode=mode)


if __name__ == '__main__':
    PickleUtils.dump_pickle("i.pkl", {1: 1})
