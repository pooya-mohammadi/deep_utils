import os
import numpy as np
from abc import abstractmethod
from deep_utils.utils.pickle_utils.pickle_utils import PickleUtils
from deep_utils.main_abs.main import MainClass
from deep_utils.utils.dict_named_tuple_utils import dictnamedtuple
from deep_utils.utils.dir_utils.dir_utils import remove_create
from deep_utils.utils.os_utils.os_path import split_extension
from deep_utils.utils.logging_utils.logging_utils import log_print

OUTPUT_CLASS = dictnamedtuple(
    "FaceRecognizer", ["encodings"])


class FaceRecognition(MainClass):
    def __init__(self, name, file_path, **kwargs):
        super().__init__(name, file_path=file_path, **kwargs)
        self.output_class = OUTPUT_CLASS
        self.normalizer = self.load_normalizer(self.config.normalizer)

    @abstractmethod
    def extract_embeddings(self, img, is_rgb, get_time=False) -> OUTPUT_CLASS:
        pass

    def extract_dir(
            self,
            image_directory,
            extensions=(".png", ".jpg", ".jpeg"),
            cropped_encoding_dir=None,
            remove_res_dir=False,
            get_mean=False,
    ):
        import cv2
        results = dict()
        remove_create(cropped_encoding_dir, remove=remove_res_dir)
        for item_name in os.listdir(image_directory):
            _, extension = os.path.splitext(item_name)
            if extension in extensions:
                img_path = os.path.join(image_directory, item_name)
                img = cv2.imread(img_path)
                result = self.extract_embeddings(img, is_rgb=False, get_time=True)
                print(f'{img_path}: time= {result["elapsed_time"]}')

                if cropped_encoding_dir and not get_mean:
                    PickleUtils.dump_pickle(os.path.join(cropped_encoding_dir, split_extension(item_name, extension=".pkl")),
                                            result.encodings)
                results[img_path] = result['encodings']
        if get_mean:

            encode = np.sum(np.array(list(results.values())), axis=0)
            encode = self.normalizer.transform(np.expand_dims(encode, axis=0))[0]
            results['mean-encoding'] = encode
            PickleUtils.dump_pickle(os.path.join(cropped_encoding_dir, "mean-encoding.pkl"), encode)
        return results

    def extract_dir_of_dir(
            self,
            input_directory,
            image_dir_name="cropped",
            encoding_dir_name="encodings",
            extensions=(".png", ".jpg", ".jpeg"),
            remove_encoding=True,
            get_mean=False,
    ):
        results = dict()
        for directory_name in sorted(os.listdir(input_directory)):
            directory_path = os.path.join(input_directory, directory_name)
            images_dir = os.path.join(directory_path, image_dir_name)
            cropped_encoding_dir = os.path.join(directory_path, encoding_dir_name)
            if not os.path.isdir(directory_path) or not os.path.isdir(images_dir):
                log_print(None, f"Skip {directory_path}...")
                continue
            remove_create(cropped_encoding_dir, remove=remove_encoding)
            dir_result = self.extract_dir(images_dir, extensions=extensions, cropped_encoding_dir=cropped_encoding_dir,
                                          remove_res_dir=remove_encoding,
                                          get_mean=get_mean)
            results[directory_name] = dir_result
        if get_mean:
            results = {k: v['mean-encoding'] for k, v in results.items()}
            encoding_name = os.path.split(input_directory)[-1]
            PickleUtils.dump_pickle(os.path.join(input_directory, encoding_name + ".pkl"), results)
        return results

    @staticmethod
    def load_normalizer(normalizer_name):

        if normalizer_name == "l2_normalizer":
            from sklearn.preprocessing import Normalizer
            l2_normalizer = Normalizer('l2')
            return l2_normalizer
        return lambda x: x