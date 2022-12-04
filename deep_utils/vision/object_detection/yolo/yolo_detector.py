import os
import shutil
import sys
import time
from abc import ABC
from os.path import join, isfile
from pathlib import Path
from typing import Dict, List, Type, Union

import numpy as np
import torch
from tqdm import tqdm

from deep_utils.main_abs import MainClass
from deep_utils.utils.box_utils.boxes import Box
from deep_utils.utils.dir_utils.dir_utils import (
    dir_train_test_split,
    remove_create,
    transfer_directory_items,
    file_incremental,
    split_extension
)
from deep_utils.utils.lib_utils.lib_decorators import (
    get_from_config,
    in_shape_fix,
    lib_rgb2bgr,
    out_shape_fix,
)
from deep_utils.utils.logging_utils import log_print
from deep_utils.utils.opencv_utils.main import show_destroy_cv2
from deep_utils.utils.os_utils.os_path import split_extension
from deep_utils.utils.dict_named_tuple_utils import dictnamedtuple
from deep_utils.utils.shutil_utils.shutil_utils import mv_or_copy

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
    def test_label_dir(dataset_dir, rename_dict: Union[Dict[int, str], None], images_name="images",
                       labels_name="labels"):
        images_path = join(dataset_dir, images_name)
        for name in sorted(os.listdir(images_path)):
            img_address = join(images_path, name)
            text_address = join(dataset_dir, labels_name, split_extension(name, '.txt'))
            YOLOObjectDetector.test_label(img_address, text_address, rename_dict=rename_dict)

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
            show_destroy_cv2(img)
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
            image_based=True
    ):
        """

        :param base_dir:
        :param out_dir:
        :param mode:
        :param test_size:
        :param remove_out_dir:
        :param skip_transfer: If the file does not exist, skip and do not raise Error
        :param remove_in_dir: if mode is mv and this is set to true the in_dir will be removed
        :param image_based: If set to True, the split will be done based on images not labels
        :return:
        """
        split_identifier_name = "images" if image_based else "labels"
        split_identifier_name_prime = "labels" if image_based else "images"
        identifier_train_names, identifier_val_names = dir_train_test_split(
            join(base_dir, split_identifier_name),
            train_dir=join(out_dir, "train", split_identifier_name),
            val_dir=join(out_dir, "val", split_identifier_name),
            mode=mode,
            remove_out_dir=remove_out_dir,
            test_size=test_size,
            remove_in_dir=remove_in_dir
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
    def rename_labels(labels_dir, rename_dict: dict):
        """
        rename labels of a directory of labels based on the input rename dictionary
        :param labels_dir:
        :param rename_dict: A dictionary by which the input labels will be renamed
        :return:
        """
        assert os.path.isdir(labels_dir), "The input is not a directory"

        for label_name in os.listdir(labels_dir):
            if label_name.endswith(".txt"):
                label_path = join(labels_dir, label_name)
                with open(label_path, mode='r') as read_file:
                    labels = []
                    for line in read_file.readlines():
                        line = line.strip()
                        label, *box = line.split(" ")
                        labels.append(" ".join([str(rename_dict.get(int(label), label)), *box]))
                with open(label_path, mode='w') as write_file:
                    write_file.writelines(labels)
