import os

from deep_utils.utils.dir_utils.dir_utils import remove_create
from deep_utils.utils.logging_utils import log_print
from deep_utils.utils.json_utils import dump_json, load_json


def get_categories(json_paths):
    """
    getting categories by updating all together.
    :param json_paths:
    :return:
    """
    categories = {}
    for json_path in json_paths:
        data = load_json(json_path)
        category = data["categories"]
        categories.update(category)
        # break
    return categories


def get_img_to_name_dict(images):
    """
    Gets {img-id: img-data} dictionary
    :param images:
    :return:
    """
    dict_ = dict()
    for img_dict in images:
        id_ = img_dict.pop("id")
        dict_[id_] = img_dict
    return dict_


def getting_img_information(json_paths, img_paths):
    all_data = []
    for json_path, img_path in zip(json_paths, img_paths):
        try:
            data = load_json(json_path)
        except:
            print(f"[INFO] {json_path} has a problem, continuing...")
            continue
        img_to_name = get_img_to_name_dict(data["images"])
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            img_info = img_to_name[img_id].copy()
            # Getting full address
            img_info["file_name"] = os.path.join(
                img_path, img_info["file_name"])
            ann["image_id"] = img_info
            all_data.append(ann)
        print(f"[INFO] {json_path} is done!")
    return all_data


def combine_jsons(images, file_list, des, json_name, json_paths):
    """
    combining jsons
    :param images:
    :param file_list:
    :param des:
    :param json_name:
    :param json_paths:
    :return:
    """
    import shutil
    from copy import deepcopy

    file_paths = [img_info[-1] for img_info in images]
    new = [ann for ann in file_list if ann["image_id"]
           ["file_name"] in file_paths]

    path_id = {
        file_pth: (en, width, height, str(en) +
                   "_" + os.path.basename(file_pth))
        for en, (width, height, file_pth) in enumerate(images, 1)
    }
    json_images = [
        {"id": id_, "width": width, "height": height, "file_name": file_name}
        for _, (id_, width, height, file_name) in path_id.items()
    ]
    annotations = []
    for en, ann in enumerate(deepcopy(new), 1):
        image_id = path_id[ann["image_id"]["file_name"]][0]
        ann["image_id"] = image_id
        ann["id"] = en
        annotations.append(ann)

    json_dict = {
        "info": {"description": "my-project-name"},
        "images": json_images,
        "annotations": annotations,
        "categories": get_categories(json_paths),
    }
    dump_json(json_name, json_dict)

    for file_pth, (en, width, height, name) in path_id.items():
        shutil.copy(file_pth, os.path.join(des, name))


def combine_coco_json(json_paths, image_dirs, result_img_dir, result_json_path):
    remove_create(result_img_dir)
    if os.path.isfile(result_json_path):
        os.remove(result_json_path)

    all_images = getting_img_information(json_paths, image_dirs)
    images = list(set([tuple(ann["image_id"].values()) for ann in all_images]))
    combine_jsons(images, all_images, result_img_dir,
                  result_json_path, json_paths)


def get_image_id(json_dict, img_name):
    images = json_dict["images"]
    for d in images:
        if d["file_name"] == img_name:
            return d["id"]


def get_annotation(json_dict, id_):
    annotations = json_dict["annotations"]
    annotations_list = list()
    for ann in annotations:
        if ann["image_id"] == id_:
            annotations_list.append(ann)
    return annotations_list


def get_coco_json_masks(json_path, mask_path, logger=None, verbose=1):
    """
    Converting json coco file segmentations to masks for Unet and other methods.
    """
    from collections import defaultdict

    import cv2
    import numpy as np
    from tqdm import tqdm

    remove_create(mask_path)
    json_dict = load_json(json_path)
    # get the annotations
    annotations_point = json_dict["annotations"]
    # make a dictionary from annotations with image_ids as keys for each of use.
    # img_annotation_dict = {ann['image_id']: ann for ann in annotations_point}
    img_annotation_dict = defaultdict(list)
    for ann in annotations_point:
        img_annotation_dict[ann["image_id"]].append(ann)
    # img_annotation_dict = {1: [{segmentations: ....}, {segmentations: ....}, ...]}
    # get the images' information
    # [{img_name: ... , width, id:... } , {}, ... ]
    image_file = json_dict["images"]
    # make a dictionary of images' information for ease of use.
    img_dict = {img_file["id"]: img_file for img_file in image_file}
    for img_id, img_info in tqdm(img_dict.items(), total=len(img_dict)):
        mask = np.zeros(
            (img_info["height"], img_info["width"]), dtype=np.uint8)
        annotations = img_annotation_dict[img_id]
        for ann in annotations:
            segmentations = ann["segmentation"]
            cat_id = ann["category_id"]
            for seg in segmentations:
                # getting the points
                xs = seg[0::2]
                ys = seg[1::2]
                pts = np.array([[x, y]
                               for x, y in zip(xs, ys)], dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))

                # draw the points on the mask image.
                try:
                    mask = cv2.fillPoly(mask, [pts], cat_id)
                except Exception as e:
                    log_print(
                        logger,
                        f"Error for {img_info['file_name']}, len={len(pts)}",
                        verbose=verbose,
                    )
        cv2.imwrite(os.path.join(mask_path, img_info["file_name"]), mask)
        log_print(logger, f"img_id: {img_id} is done!", verbose=verbose)


def create_json(images, annotations, categories, des, json_name, main_image):
    import os
    import shutil
    from copy import deepcopy

    from tqdm import tqdm

    file_paths = [img_info["id"] for img_info in images]
    new_annotations = [
        ann for ann in annotations if ann["image_id"] in file_paths]

    old_id = dict()
    new_images = deepcopy(images)
    for en, image in enumerate(new_images, 1):
        old_id[image["id"]] = en
        image["id"] = en

    annotations = []
    for ann_id, ann in enumerate(deepcopy(new_annotations), 1):
        image_id = old_id[ann["image_id"]]
        ann["image_id"] = image_id
        ann["id"] = ann_id
        annotations.append(ann)

    json_dict = {
        "info": {"description": "my-project-name"},
        "images": new_images,
        "annotations": annotations,
        "categories": categories,
    }

    dump_json(json_name, json_dict)

    for image in tqdm(new_images, total=len(new_images)):
        file_name = image["file_name"]
        shutil.copy(os.path.join(main_image, file_name),
                    os.path.join(des, file_name))


def coco_json_train_test_split(
    main_json,
    main_image,
    train_json="./train.json",
    train_image="./train",
    val_json="./val.json",
    val_image="./val",
    split_size=0.2,
):
    from sklearn.model_selection import train_test_split

    remove_create(train_image)
    remove_create(val_image)
    if os.path.isfile(val_json):
        os.remove(val_json)
    if os.path.isfile(train_json):
        os.remove(train_json)
    main_json_dict = load_json(main_json)
    images = main_json_dict["images"]
    annotations = main_json_dict["annotations"]
    categories = main_json_dict["categories"]
    train_images, val_images = train_test_split(images, test_size=split_size)
    create_json(
        train_images, annotations, categories, train_image, train_json, main_image
    )
    create_json(val_images, annotations, categories,
                val_image, val_json, main_image)


def convert_coco_json_yolo(path_to_json, img_dir, label_path, res_img):
    import shutil

    import cv2

    data = load_json(path_to_json)
    images = data["images"]
    annotations = data["annotations"]
    for image in images:
        img_id = image["id"]
        file_name = image["file_name"]
        img_path = os.path.join(img_dir, file_name)
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        bboxes = []
        for ann in annotations:
            if ann["image_id"] == img_id:
                x1, y1, w, h = ann["bbox"]
                x_c, y_c, w, h = (
                    (x1 + w / 2) / img_w,
                    (y1 + h / 2) / img_h,
                    w / img_w,
                    h / img_h,
                )
                category_id = int(ann["category_id"]) - 1
                bboxes.append([category_id, x_c, y_c, w, h])
        if bboxes:
            with open(
                os.path.join(
                    label_path, ".".join(
                        os.path.basename(file_name).split(".")[:-1])
                )
                + ".txt",
                mode="w",
            ) as f:
                for bbox_info in bboxes:
                    f.write(" ".join(str(b) for b in bbox_info))
                    f.write("\n")
            shutil.copy(img_path, res_img)
