import os
import shutil
from abc import ABC
from os.path import join, split
from typing import Dict, List, Union, Type, Literal
from deep_utils.utils.dir_utils.dir_utils import DirUtils
from tqdm import tqdm
import cv2
from deep_utils.main_abs.main import MainClass
from deep_utils.utils.box_utils.boxes import Box
from deep_utils.utils.dict_named_tuple_utils import dictnamedtuple
from deep_utils.utils.dir_utils.dir_utils import (
    dir_train_test_split,
    remove_create,
    transfer_directory_items,
    file_incremental
)
from deep_utils.utils.logging_utils.logging_utils import log_print
from deep_utils.utils.opencv_utils.opencv_utils import CVUtils
from deep_utils.utils.os_utils.os_path import split_extension
from deep_utils.utils.shutil_utils.shutil_utils import mv_or_copy
from deep_utils.utils.os_utils.os_path import validate_file_extension
from deep_utils.utils.shutil_utils.shutil_utils import mv_or_copy_list

OUTPUT_CLASS = dictnamedtuple(
    "Object", ["class_indices", "boxes", "confidences", "class_names", "elapsed_time"]
)


class OutputType:
    class_indices: List[int]
    boxes: List[List[int]]
    confidences: List[float]
    class_names: List[str]
    elapsed_time: float


class YOLOObjectDetector(MainClass, ABC):
    """
    This class contains handy functions for yolo versions!
    """

    @staticmethod
    def test_label_dir(dataset_dir, rename_dict: Union[Dict[int, str], None] = None, images_name="images",
                       labels_name="labels", show=True):
        images_path = join(dataset_dir, images_name)
        for name in sorted(os.listdir(images_path)):
            img_address = join(images_path, name)
            text_address = join(dataset_dir, labels_name, split_extension(name, '.txt'))
            if show:
                YOLOObjectDetector.test_label(img_address, text_address, rename_dict=rename_dict, show=show)
            else:
                img = YOLOObjectDetector.test_label(img_address, text_address, rename_dict=rename_dict, show=show)
                yield img

    @staticmethod
    def test_label(img_path, label_path, rename_dict: Union[Dict[int, str], None] = None, show=True):
        import cv2

        img = cv2.imread(img_path)
        boxes, texts, orgs = [], [], []
        with open(label_path, mode="r") as f:
            for line in f.readlines():
                label, xc, yc, w, h = line.strip().split()
                xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                boxes.append([xc, yc, w, h])
                if rename_dict is not None:
                    label = rename_dict.get(int(label), label)
                texts.append(f"label: {label}")
        boxes = Box.box2box(boxes,
                            in_format=Box.BoxFormat.XCYC,
                            to_format=Box.BoxFormat.XYXY,
                            in_source="CV",
                            to_source="Numpy",
                            in_relative=True,
                            to_relative=False,
                            shape_source="Numpy",
                            shape=img.shape[:2],
                            )
        img = Box.put_box_text(img, boxes, texts)
        if show:
            CVUtils.show_destroy_cv2(img)
        return img

    @staticmethod
    def split_dataset(
            base_dir,
            out_dir,
            mode="cp",
            test_size=0.2,
            remove_out_dir=False,
            skip_transfer=True,
            remove_in_dir=False,
            image_based=True,
            move_background_to_train=True,
            logger=None,
            verbose=1,
    ):
        """

        :param base_dir:
        :param out_dir:
        :param mode: cp or mv
        :param test_size:
        :param remove_out_dir:
        :param skip_transfer: If the file does not exist, skip and do not raise Error
        :param remove_in_dir: if mode is mv and this is set to true the in_dir will be removed
        :param image_based: If set to True, the split will be done based on images not labels
        :param move_background_to_train: if set to True, the images with no labels are collected and moved to train
        folder prior to other preprocessing requested
        :param logger
        :param verbose
        :return:
        """
        images_path, labels_path = join(base_dir, "images"), join(base_dir, "labels")
        if move_background_to_train:
            lbl_names_dict = {split_extension(name)[0]: join(labels_path, name) for name in os.listdir(labels_path)}
            background_images = [join(images_path, name) for name in os.listdir(images_path) if
                                 split_extension(name)[0] not in lbl_names_dict]
            background_image_names = [split(name)[-1] for name in background_images]
            # apply the remove operation
            remove_create(join(out_dir, "train", "images"), remove=remove_out_dir)
            remove_create(join(out_dir, "train", "labels"), remove=remove_out_dir)
            remove_create(join(out_dir, "val", "images"), remove=remove_out_dir)
            remove_create(join(out_dir, "val", "labels"), remove=remove_out_dir)

            # set to False in order to make sure the background images are not removed
            remove_out_dir = False
            mv_or_copy_list(background_images, join(out_dir, "train", "images"), mode=mode)
            log_print(logger, f"Moved {len(background_images)} background image to {join(out_dir, 'train', 'images')}",
                      verbose=verbose)
        else:
            background_image_names = None

        split_identifier_name = "images" if image_based else "labels"
        split_identifier_name_prime = "labels" if image_based else "images"
        identifier_train_names, identifier_val_names = dir_train_test_split(
            join(base_dir, split_identifier_name),
            train_dir=join(out_dir, "train", split_identifier_name),
            val_dir=join(out_dir, "val", split_identifier_name),
            mode=mode,
            remove_out_dir=remove_out_dir,
            test_size=test_size,
            remove_in_dir=remove_in_dir,
            ignore_list=background_image_names,
        )
        if image_based:
            identifier_train_prime = [os.path.splitext(name)[0] + ".txt" for name in identifier_train_names]
            identifier_val_prime = [os.path.splitext(name)[0] + ".txt" for name in identifier_val_names]
        else:
            img_names_dict = {split_extension(name)[0]: name for name in
                              os.listdir(join(base_dir, split_identifier_name_prime))}
            identifier_train_prime = [img_names_dict[split_extension(name)[0]] for name in identifier_train_names]
            identifier_val_prime = [img_names_dict[split_extension(name)[0]] for name in identifier_val_names]

        transfer_directory_items(
            join(base_dir, split_identifier_name_prime),
            join(out_dir, "train", split_identifier_name_prime),
            identifier_train_prime,
            mode=mode,
            remove_out_dir=remove_out_dir,
            skip_transfer=skip_transfer,
            remove_in_dir=False,  # by removing dir at this point, there would be no data for the next transfer :)
        )
        transfer_directory_items(
            join(base_dir, split_identifier_name_prime),
            join(out_dir, "val", split_identifier_name_prime),
            identifier_val_prime,
            mode=mode,
            remove_out_dir=remove_out_dir,
            skip_transfer=skip_transfer,
            remove_in_dir=remove_in_dir,
        )
        if remove_in_dir:
            shutil.rmtree(base_dir)

    @staticmethod
    def extract_label(label_path, img_path=None, shape=None, shape_source=None):
        with open(label_path, mode="r") as f:
            boxes, labels = [], []
            for line in f.readlines():
                label, b1, b2, b3, b4 = line.strip().split()
                boxes.append([float(b1), float(b2), float(b3), float(b4)])
                labels.append(int(label))

        if img_path is not None:
            import cv2
            shape = cv2.imread(img_path).shape[:2]
            shape_source = "Numpy"
        if shape is not None and shape_source is not None:
            boxes = Box.box2box(
                boxes,
                in_source=Box.BoxSource.CV,
                to_source="Numpy",
                in_format="XCYC",
                to_format="XYXY",
                in_relative=True,
                to_relative=False,
                shape=shape,
                shape_source=shape_source,
            )
        return boxes, labels

    @staticmethod
    def extract_save_label(label_path, image_path, output_path: str, remove_output_path: bool = False):
        """
        Extract box images and save them!
        :param label_path:
        :param image_path:
        :param output_path:
        :param remove_output_path:
        :return:
        """
        from deep_utils.utils.box_utils.boxes import Box
        boxes, labels = YOLOObjectDetector.extract_label(label_path, image_path)
        img = cv2.imread(image_path)
        output = Box.get_box_img(img, boxes)
        DirUtils.remove_create(output_path, remove=remove_output_path)
        output_img = join(output_path, split(image_path)[-1])
        for index, (box_img, lbl) in enumerate(zip(output, labels)):
            cv2.imwrite(DirUtils.split_extension(output_img, suffix=f"_{lbl}_{index}"), box_img)

    @staticmethod
    def get_labels_images_list(label_dir: str, image_dir: str) -> List[Dict[Literal["image", "label"], str]]:

        label_names = {DirUtils.split_extension(name)[0]: name for name in os.listdir(label_dir)}
        image_names = {DirUtils.split_extension(name)[0]: name for name in os.listdir(image_dir)}
        output = [{"label": join(label_dir, name), "image": join(image_dir, image_names[key])} for key, name in
                  label_names.items() if key in image_names]
        return output

    @staticmethod
    def extract_save_directory(label_dir: str, image_dir: str, output_path, remove_output_path: bool = True):
        """
        Extracts image boxes of a directory
        :param label_dir:
        :param image_dir:
        :param output_path:
        :param remove_output_path:
        :return:
        """
        data = YOLOObjectDetector.get_labels_images_list(label_dir, image_dir)
        DirUtils.remove_create(output_path, remove=remove_output_path)
        for pair in data:
            label_path = pair['label']
            image_path = pair['image']
            YOLOObjectDetector.extract_save_label(label_path, image_path, output_path,
                                                  remove_output_path=False)

    @staticmethod
    def clean_samples(label_path, img_path, logger=None, verbose=1):
        """
        Using this method, one can remove images and the labels do not have their correspondences.
        :param label_path:
        :param img_path:
        :param logger:
        :param verbose:
        :return:
        """
        c = 0
        image_names, image_extensions = [], []
        for img in os.listdir(img_path):
            image_name, image_ext = os.path.splitext(img)
            image_names.append(image_name)
            image_extensions.append(image_ext)

        label_names = [os.path.splitext(l)[0] for l in os.listdir(label_path)]
        for label in label_names:
            if label not in image_names:
                remove_lbl_path = join(label_path, label + ".txt")
                log_print(logger, f"Removed {remove_lbl_path}", verbose=verbose)
                os.remove(remove_lbl_path)
                c += 1

        for img_name, img_ext in zip(image_names, image_extensions):
            if img_name not in label_names:
                remove_img_path = join(img_path, img_name + img_ext)
                log_print(logger, f"Removed {remove_img_path}", verbose=verbose)
                os.remove(remove_img_path)
                c += 1

        log_print(logger, f"Removed {c} samples!", verbose=verbose)

    @staticmethod
    def combine_datasets(
            dataset_paths: List[str],
            final_dataset_path: str,
            remove_final_dataset: bool = False,
            extra_index_format="prefix"
    ):
        """

        :param dataset_paths:
        :param final_dataset_path:
        :param remove_final_dataset:
        :param extra_index_format: whether to add suffix, prefix, or nothing as an extra index to data
        :return:
        """
        remove_create(final_dataset_path, remove_final_dataset)

        for dataset_path in dataset_paths:
            YOLOObjectDetector.transfer_dataset(dataset_path, final_dataset_path, extra_index_format)

    @staticmethod
    def transfer_dataset(dataset_path, final_dataset_path, extra_index_format="prefix", mode="cp", logger=None,
                         verbose=1):
        """
        This method is used to transfer samples from a dataset to other one
        :param dataset_path:
        :param final_dataset_path:
        :param extra_index_format:
        :param mode: Whether copy or move the dataset
        :param logger:
        :param verbose:
        :return:
        """
        images_path = os.path.join(dataset_path, "images")
        labels_path = os.path.join(dataset_path, "labels")

        final_images = os.path.join(final_dataset_path, "images")
        final_labels = os.path.join(final_dataset_path, "labels")

        os.makedirs(final_images, exist_ok=True)
        os.makedirs(final_labels, exist_ok=True)

        if not os.path.isdir(images_path) or not os.path.isdir(labels_path):
            log_print(logger, f"{images_path} or {labels_path} do not exit!", verbose=verbose)
            return

        # to make sure it's not replicated
        # index = len(os.listdir(final_images)) + len(os.listdir(final_labels))
        img_dict = dict()
        for img_name in tqdm(
                os.listdir(images_path),
                total=len(os.listdir(images_path)),
                desc=f"copying {images_path} to {final_images}",
        ):
            img_path = join(images_path, img_name)
            # new_name = split_extension(img_name, artifact_type=extra_index_format, artifact_value=img_index)
            new_path = join(final_images, img_name)
            new_path = file_incremental(new_path, artifact_type=extra_index_format)
            mv_or_copy(img_path, new_path, mode=mode)
            # key: original name without extension, val: new name without extension
            img_dict[split_extension(img_name)[0]] = split_extension(os.path.split(new_path)[-1])[0]

        for lbl_img_bare_name in tqdm(
                os.listdir(labels_path),
                total=len(os.listdir(labels_path)),
                desc=f"copying {labels_path} to {final_labels}"):
            lbl_path = os.path.join(labels_path, lbl_img_bare_name)
            lbl_bare_name = split_extension(lbl_img_bare_name)[0]
            lbl_img_bare_name = img_dict.get(lbl_bare_name, None)
            if lbl_img_bare_name is None:
                log_print(logger, f"label: {lbl_img_bare_name} has no image counterpart", verbose=verbose)
                continue
            mv_or_copy(lbl_path, os.path.join(final_labels, lbl_img_bare_name + ".txt"), mode=mode)
        log_print(logger, f"Successfully {mode}-ed {dataset_path} to {final_dataset_path}")

    @staticmethod
    def rename_labels(labels_dir, rename_dict: dict, output_dir: str = None):
        """
        rename labels of a directory of labels based on the input rename dictionary
        :param labels_dir:
        :param rename_dict: A dictionary by which the input labels will be renamed
        :param output_dir: if provided, the labels are created in the provided directory
        :return:
        """
        assert os.path.isdir(labels_dir), "The input is not a directory"

        output_dir = labels_dir if output_dir is None else output_dir
        for label_name in os.listdir(labels_dir):
            if label_name.endswith(".txt"):
                with open(join(labels_dir, label_name), mode='r') as read_file:
                    labels = []
                    for line in read_file.readlines():
                        line = line.strip()
                        label, *box = line.split(" ")
                        labels.append(" ".join([str(rename_dict.get(int(label), label)), *box]))
                with open(join(output_dir, label_name), mode='w') as write_file:
                    for line in labels:
                        write_file.write(line + "\n")

    def detect_objects(self,
                       img,
                       is_rgb,
                       class_indices=None,
                       confidence=None,
                       iou_thresh=None,
                       img_size=None,
                       agnostic=None,
                       get_time=False,
                       logger=None,
                       verbose=1) -> Union[Type[OutputType], Dict[str, list]]:
        raise NotImplementedError("object_detects is not implemented!")

    def detect_img_file(self,
                        img_file,
                        class_indices=None,
                        confidence=None,
                        iou_thresh=None,
                        img_size=None,
                        agnostic=None,
                        get_time=False,
                        logger=None,
                        verbose=1) -> Union[Type[OutputType], Dict[str, list]]:
        img = cv2.imread(img_file)
        if img is None:
            log_print(logger, f"Image {img_file} is not valid", verbose=verbose)
        else:
            return self.detect_objects(img, False, class_indices, confidence,
                                       iou_thresh, img_size, agnostic, get_time,
                                       logger, verbose)

    def detect_dir(self,
                   dir_,
                   confidence=None,
                   iou_thresh=None,
                   classes=None,
                   extensions=(".png", ".jpg", ".jpeg"),
                   save_labels_dir=None,
                   remove_labels_dir=True,
                   cut_images_dir=None,
                   remove_cut_images_dir=True,
                   create_labels=False,
                   logger=None,
                   verbose=1) -> Union[Dict[str, Type[OutputType]]]:
        import cv2
        outputs = {}
        images = {}
        for file_name in os.listdir(dir_):
            if validate_file_extension(file_name, extensions):
                file_path = os.path.join(dir_, file_name)
                img = cv2.imread(file_path)
                if os.path.isfile(file_path) and img is not None:
                    output = self.detect_objects(img, is_rgb=False, class_indices=classes, iou_thresh=iou_thresh,
                                                 confidence=confidence)
                    outputs[file_name] = output
                    images[file_name] = img
                    log_print(logger, f"Successfully detected {file_path}")

        if save_labels_dir:
            remove_create(save_labels_dir, remove=remove_labels_dir)
            for file_name, output in outputs.items():
                label_path = os.path.join(save_labels_dir, split_extension(file_name, extension=".txt"))
                with open(label_path, mode="w") as f:
                    for cls_index, box in zip(output.class_indices, output.boxes):
                        box = Box.box2box(box,
                                          in_format=Box.BoxFormat.XYXY,
                                          to_format=Box.BoxFormat.XCYC,
                                          in_source="Numpy",
                                          to_source="CV",
                                          in_relative=False,
                                          to_relative=True,
                                          shape_source="Numpy",
                                          shape=images[file_name].shape[:2],
                                          )
                        f.write(f"{cls_index} {' '.join(str(b) for b in box)}\n")
                        log_print(logger, f"Successfully saved label {label_path}")
        if save_labels_dir and create_labels:
            with open(os.path.join(save_labels_dir, "labels.txt"), mode='w') as f:
                for name in self.config.class_names:
                    f.write(f"{name}\n")

        if cut_images_dir:
            remove_create(cut_images_dir, remove=remove_cut_images_dir)
            for file_name, output in outputs.items():
                for en, (cls_index, box) in enumerate(zip(output.class_indices, output.boxes)):
                    img = Box.get_box_img(images[file_name], box)
                    name, extension = split_extension(file_name)
                    cut_image_save_dir = os.path.join(cut_images_dir,
                                                      f"{name}{self.config.class_names[cls_index]}_{en}{extension}")
                    cv2.imwrite(cut_image_save_dir, img)
                    log_print(logger, f"Successfully saved cut image: {cut_image_save_dir}")

        return outputs
