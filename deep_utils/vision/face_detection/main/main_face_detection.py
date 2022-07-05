import os
from abc import abstractmethod
from deep_utils.main_abs.main import MainClass
from deep_utils.utils.dict_named_tuple_utils import dictnamedtuple
from deep_utils.utils.dir_utils.dir_utils import remove_create
from deep_utils.utils.box_utils.boxes import Box
from deep_utils.utils.os_utils.os_path import split_extension
from deep_utils.utils.logging_utils.logging_utils import log_print

OUTPUT_CLASS = dictnamedtuple(
    "FaceDetector", ["boxes", "landmarks", "confidences"])


class FaceDetector(MainClass):
    def __init__(self, name, file_path, **kwargs):
        super().__init__(name, file_path=file_path, **kwargs)
        self.output_class = OUTPUT_CLASS

    @abstractmethod
    def detect_faces(self, img, is_rgb, confidence=None, get_time=False) -> OUTPUT_CLASS:
        pass

    def detect_crop_dir(
            self,
            image_directory,
            confidence=None,
            extensions=(".png", ".jpg", ".jpeg"),
            res_dir=None,
            remove_res_dir=False,
            get_biggest=False
    ):
        import cv2
        results = dict()
        remove_create(res_dir, remove=remove_res_dir)
        for item_name in os.listdir(image_directory):
            _, extension = os.path.splitext(item_name)
            if extension in extensions:
                img_path = os.path.join(image_directory, item_name)
                img = cv2.imread(img_path)
                result = self.detect_faces(
                    img,
                    is_rgb=False,
                    confidence=confidence,
                    get_time=True,
                )
                print(
                    f'{img_path}: objects= {len(result["boxes"])}, time= {result["elapsed_time"]}'
                )
                boxes = result["boxes"]
                if get_biggest:
                    boxes = [Box.get_biggest(boxes)]
                cropped_images = Box.get_box_img(img, boxes)
                result = dict(result)
                result['cropped_images'] = cropped_images
                if res_dir:
                    for e, cropped_image in enumerate(cropped_images):
                        cv2.imwrite(os.path.join(res_dir, split_extension(item_name, suffix=f"_{e}")), cropped_image)
                results[img_path] = result
        return results

    def detect_crop_dir_of_dir(
            self,
            input_directory,
            image_dir_name="images",
            cropped_dir_name="cropped",
            confidence=None,
            extensions=(".png", ".jpg", ".jpeg"),
            remove_cropped=True,
            get_biggest=False
    ):
        for directory_name in sorted(os.listdir(input_directory)):

            directory_path = os.path.join(input_directory, directory_name)
            images_dir = os.path.join(directory_path, image_dir_name)
            cropped_dir = os.path.join(directory_path, cropped_dir_name)
            if not os.path.isdir(directory_path) or not os.path.isdir(images_dir):
                log_print(None, f"Skip {directory_path}...")
                continue
            remove_create(cropped_dir, remove=remove_cropped)
            self.detect_crop_dir(images_dir, confidence=confidence, extensions=extensions, res_dir=cropped_dir,
                                 remove_res_dir=remove_cropped, get_biggest=get_biggest)
