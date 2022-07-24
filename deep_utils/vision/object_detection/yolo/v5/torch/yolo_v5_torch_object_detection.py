import os
import shutil
import sys
import time
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

from .config import Config

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
from .models.experimental import attempt_load
from .utils_.datasets import letterbox
from .utils_.general import non_max_suppression, scale_coords

OUTPUT_CLASS = dictnamedtuple(
    "Object", ["class_indices", "boxes", "confidences", "class_names", "elapsed_time"]
)


class OutputType:
    class_indices: List[int]
    boxes: List[List[int]]
    confidences: List[float]
    class_names: List[str]
    elapsed_time: float


class YOLOV5TorchObjectDetector(MainClass):
    def __init__(
            self,
            class_names=None,
            model_weight=None,
            device="cpu",
            img_size=(320, 320),
            confidence=0.4,
            iou_thresh=0.45,
            **kwargs,
    ):
        super(YOLOV5TorchObjectDetector, self).__init__(
            name=self.__class__.__name__,
            file_path=__file__,
            class_names=class_names,
            model_weight=model_weight,
            device=device,
            img_size=img_size,
            confidence=confidence,
            iou_thresh=iou_thresh,
            **kwargs,
        )
        self.config: Config

    @staticmethod
    def yolo_resize(
            img,
            new_shape=(640, 640),
            color=(114, 114, 114),
            auto=True,
            scaleFill=False,
            scaleup=True,
    ):

        return letterbox(
            img,
            new_shape=new_shape,
            color=color,
            auto=auto,
            scaleFill=scaleFill,
            scaleup=scaleup,
        )

    def load_model(self):
        # if self.config.model_weight:
        self.model = attempt_load(
            self.config.model_weight, map_location=self.config.device
        )
        self.model.to(self.config.device)
        self.model.eval()
        img = torch.zeros((1, 3, *self.config.img_size), device=self.config.device)
        self.model(img)
        print(f"{self.name}: weights are loaded")

    def detect_objects(
            self,
            img,
            is_rgb,
            class_indices=None,
            confidence=None,
            iou_thresh=None,
            img_size=None,
            agnostic=None,
            get_time=False,
    ) -> Union[Type[OutputType], Dict[str, list]]:
        """

        :param img:
        :param is_rgb: Is used with rgb2bgr. The required conversion is done automatically.
        :param confidence:
        :param iou_thresh:
        :param class_indices: target class indices, the rest will be ignored!
        :param agnostic:
        :param get_time:
        :param img_size:
        :param kwargs:
        :return:
        """

        tic = time.time() if get_time else 0

        self.update_config(
            confidence=confidence, iou_thresh=iou_thresh, img_size=img_size
        )
        img = lib_rgb2bgr(img, target_type="rgb", is_rgb=is_rgb)
        img = in_shape_fix(img, size=4)

        im0 = img
        img = np.array(
            [self.yolo_resize(im, new_shape=self.config.img_size)[0] for im in im0]
        )
        img = img.transpose((0, 3, 1, 2))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.config.device)
        img = img / 255.0
        with torch.no_grad():
            prediction = self.model(img, augment=False)[0]
        prediction = non_max_suppression(
            prediction,
            self.config.confidence,
            self.config.iou_thresh,
            classes=class_indices,
            agnostic=agnostic,
        )
        boxes, class_names, classes, confidences = [
            [[] for _ in range(im0.shape[0])] for _ in range(4)
        ]
        for i, det in enumerate(prediction):  # detections per image
            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0[i].shape
                ).round()
                for *xyxy, conf, cls in reversed(det):
                    bbox = Box.box2box(
                        xyxy,
                        in_source=Box.BoxSource.Torch,
                        to_source=Box.BoxSource.Numpy,
                        return_int=True,
                    )
                    boxes[i].append(bbox)
                    confidences[i].append(round(conf.item(), 2))
                    cls = int(cls.item())
                    classes[i].append(cls)
                    class_names[i].append(self.config.class_names[cls])

        if get_time:
            toc = time.time()
            elapsed_time = toc - tic
        else:
            elapsed_time = 0

        output = OUTPUT_CLASS(
            class_indices=classes,
            boxes=boxes,
            confidences=confidences,
            class_names=class_names,
            elapsed_time=elapsed_time,
        )
        output = out_shape_fix(output)
        return output

    @staticmethod
    def test_label_dir(dataset_dir, rename_dict: Union[Dict[int, str], None], images_name="images",
                       labels_name="labels"):
        images_path = join(dataset_dir, images_name)
        for name in sorted(os.listdir(images_path)):
            img_address = join(images_path, name)
            text_address = join(dataset_dir, labels_name, split_extension(name, '.txt'))
            YOLOV5TorchObjectDetector.test_label(img_address, text_address, rename_dict=rename_dict)

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
    ):
        img_train_names, img_val_names = dir_train_test_split(
            join(base_dir, "images"),
            train_dir=join(out_dir, "train", "images"),
            val_dir=join(out_dir, "val", "images"),
            mode=mode,
            remove_out_dir=remove_out_dir,
            test_size=test_size,
            remove_in_dir=remove_in_dir
        )
        img_train_labels = [os.path.splitext(name)[0] + ".txt" for name in img_train_names]
        img_val_labels = [os.path.splitext(name)[0] + ".txt" for name in img_val_names]

        transfer_directory_items(
            join(base_dir, "labels"),
            join(out_dir, "train", "labels"),
            img_train_labels,
            mode=mode,
            remove_out_dir=remove_out_dir,
            skip_transfer=skip_transfer,
            remove_in_dir=False,  # by removing dir at this point, there would be no data for the next transfer :)
        )
        transfer_directory_items(
            join(base_dir, "labels"),
            join(out_dir, "val", "labels"),
            img_val_labels,
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

    @get_from_config
    def detect_dir(
            self,
            input_dir,
            confidence=None,
            iou_thresh=None,
            classes=None,
            agnostic=None,
            img_size=None,
            extensions=(".png", ".jpg", ".jpeg"),
            res_img_dir=None,
            res_label_dir=None,
            put_annotations=True,
            save_crop=False,
            remove_dirs=False,
    ):
        """

        :param input_dir:
        :param confidence:
        :param iou_thresh:
        :param classes:
        :param agnostic:
        :param img_size:
        :param extensions:
        :param res_img_dir:
        :param res_label_dir:
        :param put_annotations:
        :param save_crop: If set True, ignores put_annotations
        :param remove_dirs:
        :return:
        """
        import cv2

        results = dict()
        remove_create(res_label_dir, remove_dirs)
        remove_create(res_img_dir, remove_dirs)
        for item_name in os.listdir(input_dir):
            _, extension = os.path.splitext(item_name)
            if extension in extensions:
                img_path = os.path.join(input_dir, item_name)
                img = cv2.imread(img_path)
                result = self.detect_objects(
                    img,
                    is_rgb=False,
                    confidence=confidence,
                    iou_thresh=iou_thresh,
                    class_indices=classes,
                    agnostic=agnostic,
                    img_size=img_size,
                    get_time=True,
                )
                results[img_path] = result
                print(
                    f'{img_path}: objects= {len(result["boxes"])}, time= {result["elapsed_time"]}'
                )
                boxes = result.boxes
                if len(boxes):
                    if res_img_dir:
                        res_path = os.path.join(res_img_dir, item_name)
                        if save_crop:
                            images = Box.get_box_img(img, boxes)
                            for label, img in zip(result.class_names, images):
                                res_path = os.path.join(res_img_dir, f"{label}_{item_name}")
                                cv2.imwrite(file_incremental(res_path), img)
                        elif put_annotations:
                            img = Box.put_box_text(img, boxes, [
                                f"{name}_{conf}"
                                for name, conf in zip(
                                    result["class_names"], result["confidences"]
                                )
                            ])
                            # img = Box.put_box(img, boxes)
                            # img = Box.put_text(
                            #     img,
                            #     text=[
                            #         f"{name}_{conf}"
                            #         for name, conf in zip(
                            #             result["class_names"], result["confidences"]
                            #         )
                            #     ],
                            #     org=[(b[0], b[1]) for b in result["boxes"]],
                            # )
                            cv2.imwrite(res_path, img)
                        else:
                            cv2.imwrite(res_path, img)
                    if res_label_dir:
                        res_path = os.path.join(
                            res_label_dir, split_extension(item_name, ".txt")
                        )
                        xcyc_boxes = Box.box2box(
                            boxes,
                            in_format="XYXY",
                            to_format="XCYC",
                            in_source="Numpy",
                            to_source="CV",
                            in_relative=False,
                            to_relative=True,
                            shape_source="Numpy",
                            shape=img.shape[:2],
                        )
                        with open(res_path, mode="w") as f:
                            for (b1, b2, b3, b4), class_ in zip(
                                    xcyc_boxes, result["classes"]
                            ):
                                f.write(f"{class_} {b1} {b2} {b3} {b4}\n")

        return results

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
            YOLOV5TorchObjectDetector.transfer_dataset(dataset_path, final_dataset_path, extra_index_format)

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
