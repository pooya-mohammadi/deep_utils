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


PickleUtils.load_pickle = method_deprecated(PickleUtils.load)
PickleUtils.dump_pickle = method_deprecated(PickleUtils.dump)
