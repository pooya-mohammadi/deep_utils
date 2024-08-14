from enum import Enum
from typing import Sequence, Union, Tuple, List

import numpy as np

from deep_utils.utils.logging_utils.logging_utils import value_error_log


class Point:
    class PointSource(Enum):
        Torch = "Torch"
        TF = "TF"
        CV = "CV"
        Numpy = "Numpy"

    @staticmethod
    def point2point(
            point,
            in_source: Union[str, PointSource],
            to_source: Union[str, PointSource],
            in_relative=None,
            to_relative=None,
            shape=None,
            shape_source=None,
    ):
        """
        >>> #The ability to write in_source and to_source in any mode (capital or small letters, etc.)

        >>> Point.point2point(point = [1, 5], in_source=Point.PointSource.Numpy, to_source=Point.PointSource.Torch)
        (5, 1)
        >>> Point.point2point(point = [1, 5], in_source="nUmpy", to_source=Point.PointSource.Torch)
        (5, 1)

        >>>  #Convert point from in_relative to to_relative

        >>> Point.point2point(point = [0.1, 0.05], shape = [10,100], in_relative = False, to_relative = True, in_source = 'numpy', to_source = 'NUMPY', shape_source = Point.PointSource.Numpy)
        Traceback (most recent call last):
         Point._point2point(point = [0.1, 0.05], shape=[10,100],in_relative=False,to_relative=True, in_source='numpy', to_source='NUMPY', shape_source=Point.PointSource.Numpy)
        ValueError: the input is  relative while in_relative is set to False
        >>> Point.point2point(point = [0.1, 0.05], shape = [10,100], in_relative = True, to_relative = False, in_source = 'TF', to_source = 'tF', shape_source = Point.PointSource.Numpy)
        [1.0, 5.0]
        >>> Point.point2point(point = [1, 5], shape = [10,100], in_relative = True, to_relative=False, in_source = 'numPY', to_source = 'NUMPy', shape_source = Point.PointSource.Numpy)
        Traceback (most recent call last):
         Point._point2point(point = [1, 5], shape = [10,100], in_relative = True, to_relative = False, in_source = 'numPY', to_source = 'NUMPy', shape_source = Point.PointSource.Numpy)
        ValueError: the input is not relative while in_relative is set to True
        >>> Point.point2point(point = [1, 5], shape = [10,100], in_relative = False, to_relative = True, in_source = 'NUMPY', to_source = 'NUmpy', shape_source = Point.PointSource.Numpy)
        [0.1, 0.05]
        """
        if point is None or len(point) == 0:
            pass
        elif isinstance(point[0], (tuple, list, np.ndarray)):
            point = [
                Point._point2point(
                    p,
                    in_source=in_source,
                    to_source=to_source,
                    in_relative=in_relative,
                    to_relative=to_relative,
                    shape=shape,
                    shape_source=shape_source,
                )
                for p in point
            ]
        else:
            point = Point._point2point(
                point,
                in_source=in_source,
                to_source=to_source,
                in_relative=in_relative,
                to_relative=to_relative,
                shape=shape,
                shape_source=shape_source,
            )
        return point

    @staticmethod
    def _point2point(
            point,
            in_source: Union[str, PointSource],
            to_source: Union[str, PointSource],
            in_relative=None,
            to_relative=None,
            shape=None,
            shape_source=None,
    ):
        if isinstance(in_source, Point.PointSource):
            in_source = in_source.value
        elif isinstance(in_source, str):
            in_source = in_source
        else:
            raise Exception(
                f"in_source: {in_source} is not supported, provide Point.PointSource or equivalent string"
            )
        if isinstance(to_source, Point.PointSource):
            to_source = to_source.value
        elif isinstance(to_source, str):
            to_source = to_source
        else:
            raise Exception(
                f"in_source: {in_source} is not supported, provide Point.PointSource or equivalent string"
            )

        in_source = in_source.lower()
        to_source = to_source.lower()

        if (
                in_source in [Point.PointSource.Torch.value.lower(),
                              Point.PointSource.CV.value.lower()]
                and to_source in [Point.PointSource.TF.value.lower(), Point.PointSource.Numpy.value.lower()]
        ) or (
                in_source in [Point.PointSource.TF.value.lower(),
                              Point.PointSource.Numpy.value.lower()]
                and to_source in [Point.PointSource.Torch.value.lower(), Point.PointSource.CV.value.lower()]
        ):
            point = (point[1], point[0])
        elif (
                (in_source is None and to_source is None)
                or in_source == to_source
                or (
                        in_source in [Point.PointSource.Torch.value.lower(),
                                      Point.PointSource.CV.value.lower()]
                        and to_source
                        in [Point.PointSource.CV.value, Point.PointSource.Torch.value.lower()]
                )
                or (
                        in_source in [Point.PointSource.TF.value.lower(),
                                      Point.PointSource.Numpy.value.lower()]
                        and to_source
                        in [Point.PointSource.TF.value.lower(), Point.PointSource.Numpy.value.lower()]
                )
        ):
            pass
        else:
            raise Exception(
                f"Conversion from {in_source} to {to_source} is not Supported."
                f" Supported types: {Box._get_enum_names(Point.PointSource)}"
            )
        if to_source is not None and shape_source is not None and shape is not None:
            img_w, img_h = Point.point2point(
                shape, in_source=shape_source, to_source=to_source
            )
            if not in_relative:
                if isinstance(point[0], float) or isinstance(point[1], float):
                    raise ValueError(f"the input is  relative while in_relative is set to {in_relative}")
            if in_relative:
                if isinstance(point[0], int) or isinstance(point[1], int):
                    raise ValueError(f"the input is not relative while in_relative is set to {in_relative}")
            if not in_relative and to_relative:
                p1, p2 = point
                point = [p1 / img_w, p2 / img_h]
            elif in_relative and not to_relative:
                p1, p2 = point
                point = [p1 * img_w, p2 * img_h]
        return point

    @staticmethod
    def _put_point(
            img,
            point,
            radius,
            color=(0, 255, 0),
            thickness=None,
            lineType=None,
            shift=None,
            in_source="Numpy",
    ):
        import cv2

        if not isinstance(point, int):
            point = (int(point[0]), int(point[1]))
        point = Point.point2point(point, in_source=in_source, to_source="CV")
        return cv2.circle(img, point, radius, color, thickness, lineType, shift)

    @staticmethod
    def put_point(
            img,
            point,
            radius,
            color=(0, 255, 0),
            thickness=None,
            lineType=None,
            shift=None,
            in_source="Numpy",
    ):
        if point is None or len(point) == 0:
            pass
        elif isinstance(point[0], (tuple, list, np.ndarray)):
            for p in point:
                img = Point._put_point(
                    img, p, radius, color, thickness, lineType, shift, in_source
                )
        else:
            img = Point._put_point(
                img, point, radius, color, thickness, lineType, shift, in_source
            )
        return img

    @staticmethod
    def sort_points(pts: Union[list, tuple]):
        """
        Sort a list of 4 points based on upper-left, upper-right, down-right, down-left
        :param pts:
        :return:
        """
        top_points = sorted(pts, key=lambda l: l[0])[:2]
        top_left = min(top_points, key=lambda l: l[1])
        top_right = max(top_points, key=lambda l: l[1])
        pts.remove(top_left)
        pts.remove(top_right)
        down_left = min(pts, key=lambda l: l[1])
        down_right = max(pts, key=lambda l: l[1])
        return top_left, top_right, down_right, down_left

    @staticmethod
    def rotate_point(target_p: tuple[int, int], center_p: tuple[int, int], degree: float):
        """
        :param target_p:
        :param center_p:
        :param degree:
        :return:
        """
        degree = - degree  # to make it aligned with cv2 rotation direction
        rad = degree * np.pi / 180
        t_x, t_y = target_p
        c_x, c_y = center_p
        delta_x = (t_x - c_x)
        delta_y = (t_y - c_y)
        x = np.cos(rad) * delta_x - np.sin(rad) * delta_y + c_x
        y = np.sin(rad) * delta_x + np.cos(rad) * delta_y + c_y
        return int(x), int(y)


class Box:
    class BoxFormat(Enum):
        XYWH = "XYWH"
        XYXY = "XYXY"
        XCYC = "XCYC"

    class BoxSource(Enum):
        Torch = "Torch"
        TF = "TF"
        CV = "CV"
        Numpy = "Numpy"

    class OutType(Enum):
        Numpy = np.array
        List = list
        Tuple = tuple

    PointSource = Point.PointSource

    @staticmethod
    def check_overlap(box_a: Union[Tuple[int, int, int, int], List[int]],
                      box_b: Union[Tuple[int, int, int, int], List[int]]):
        """
        Check if two boxes overlap
        :param box_a:
        :param box_b:
        :return:
        """
        p1_x1, p1_y1, p1_x2, p1_y2 = box_a
        p2_x1, p2_y1, p2_x2, p2_y2 = box_b

        if p1_x1 > p2_x2 or p1_x2 < p2_x1:
            return False
        if p1_y1 > p2_y2 or p1_y2 < p2_y1:
            return False
        return True

    @staticmethod
    def resize_box(box: List[int],
                   img_input_shape: Union[list, tuple] = None,
                   img_resized_shape: Union[list, tuple] = None,
                   return_int: bool = True):
        """
        Resize a box from one source to another when the image is resized. Box and shapes must be in Numpy source.
        Box must be in XYXY format.
        :param box:
        :param img_input_shape:
        :param img_resized_shape:
        :param return_int: if True, return the box as int, else return as float
        :return:
        """
        if box is None or len(box) == 0:
            raise ValueError("box is None or empty")
        if len(box) != 4:
            raise ValueError(f"box must have 4 elements, got {len(box)}")
        # get the ratio
        if img_input_shape is None or img_resized_shape is None:
            raise ValueError("img_input_shape and img_resized_shape must be provided")
        if len(img_input_shape) != 2 or len(img_resized_shape) != 2:
            raise ValueError("img_input_shape and img_resized_shape must be 2D")
        ratio_w = img_resized_shape[0] / img_input_shape[0]
        ratio_h = img_resized_shape[1] / img_input_shape[1]
        # resize the box
        resized_box = [
            box[0] * ratio_w,
            box[1] * ratio_h,
            box[2] * ratio_w,
            box[3] * ratio_h,
        ]
        if return_int:
            resized_box = [round(b) for b in resized_box]
        return resized_box

    @staticmethod
    def box2box(
            box,
            in_format=None,
            to_format=None,
            in_source: Union[str, BoxSource] = BoxSource.Numpy,
            to_source: Union[str, BoxSource] = BoxSource.Numpy,
            in_relative=None,
            to_relative=None,
            shape=None,
            shape_source: Union[str, Point.PointSource] = None,
            out_type=None,
            return_int=None,
    ):
        if box is None or len(box) == 0:
            pass
        elif isinstance(box[0], (tuple, list, np.ndarray)):
            box = [
                Box._box2box(
                    b,
                    in_format=in_format,
                    to_format=to_format,
                    in_source=in_source,
                    to_source=to_source,
                    in_relative=in_relative,
                    to_relative=to_relative,
                    shape=shape,
                    shape_source=shape_source,
                    out_type=out_type,
                    return_int=return_int,
                )
                for b in box
            ]

        else:
            box = Box._box2box(
                box,
                in_format=in_format,
                to_format=to_format,
                in_source=in_source,
                to_source=to_source,
                in_relative=in_relative,
                to_relative=to_relative,
                shape=shape,
                shape_source=shape_source,
                out_type=out_type,
                return_int=return_int,
            )
        return box

    @staticmethod
    def _box2box(
            box,
            in_format=None,
            to_format=None,
            in_source=None,
            to_source=None,
            in_relative=None,
            to_relative=None,
            shape=None,
            shape_source: Point.PointSource = None,
            out_type=None,
            return_int=None,
            logger=None
    ):
        """

        :param box:
        :param in_format:
        :param to_format:
        :param in_source:
        :param to_source:
        :param in_relative:
        :param shape:
        :param to_relative:
        :param out_type: output type of the box. Supported types: list, tuple, numpy
        :param logger:
        :return:
        """
        if isinstance(in_format, Box.BoxFormat):
            in_format = in_format.value
        if isinstance(to_format, Box.BoxFormat):
            to_format = to_format.value

        if isinstance(in_source, Box.BoxSource):
            in_source = in_source.value
        if isinstance(to_source, Box.BoxSource):
            to_source = to_source.value
        for b in box:
            if b < 0:
                raise ValueError("Box values cannot be negative values")
        if (
                in_format == Box.BoxFormat.XYWH.value
                and to_format == Box.BoxFormat.XYXY.value
        ):
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            box = [x1, y1, x2, y2]
        elif (
                in_format == Box.BoxFormat.XYXY.value
                and to_format == Box.BoxFormat.XYWH.value
        ):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            box = [x1, y1, w, h]
        elif (
                in_format == Box.BoxFormat.XYXY.value
                and to_format == Box.BoxFormat.XCYC.value
        ):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            box = [xc, yc, w, h]
        elif (
                in_format == Box.BoxFormat.XCYC.value
                and to_format == Box.BoxFormat.XYXY.value
        ):
            xc, yc, w, h = box
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            box = [x1, y1, x2, y2]
        elif (
                in_format == Box.BoxFormat.XYWH.value
                and to_format == Box.BoxFormat.XCYC.value
        ):
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            box = [xc, yc, w, h]
        elif (
                in_format == Box.BoxFormat.XCYC.value
                and to_format == Box.BoxFormat.XYWH.value
        ):
            xc, yc, w, h = box
            x1, y1 = xc - w // 2, yc - h // 2

            box = [x1, y1, w, h]
        elif (in_format is None and to_format is None) or in_format == to_format:
            pass
        else:
            raise Exception(
                f"Conversion form {in_format} to {to_format} is not Supported."
                f" Supported types: {Box._get_enum_names(Box.BoxFormat)}"
            )

        if (
                in_source in [Box.BoxSource.Torch.value, Box.BoxSource.CV.value]
                and to_source in [Box.BoxSource.TF.value, Box.BoxSource.Numpy.value]
        ) or (
                in_source in [Box.BoxSource.TF.value, Box.BoxSource.Numpy.value]
                and to_source in [Box.BoxSource.Torch.value, Box.BoxSource.CV.value]
        ):
            box = [box[1], box[0], box[3], box[2]]
        elif (
                (in_source is None and to_source is None)
                or in_source == to_source
                or (
                        in_source in [Box.BoxSource.Torch.value,
                                      Box.BoxSource.CV.value]
                        and to_source in [Box.BoxSource.CV.value, Box.BoxSource.Torch.value]
                )
                or (
                        in_source in [Box.BoxSource.TF.value,
                                      Box.BoxSource.Numpy.value]
                        and to_source in [Box.BoxSource.TF.value, Box.BoxSource.Numpy.value]
                )
        ):
            pass
        else:
            raise Exception(
                f"Conversion form {in_source} to {to_source} is not Supported."
                f" Supported types: {Box._get_enum_names(Box.BoxSource)}"
            )
        if in_relative != to_relative:
            if to_source is not None and shape_source is not None and shape is not None:
                img_w, img_h = Point.point2point(
                    shape, in_source=shape_source, to_source=to_source
                )
                if not in_relative and to_relative:
                    b1, b2, b3, b4 = box
                    if b1 > img_w or b2 > img_h or b3 > img_w or b4 > img_h:
                        raise ValueError(f"box values:{box} cannot be larger than shape:{img_w, img_h}")
                    box = [b1 / img_w, b2 / img_h, b3 / img_w, b4 / img_h]
                elif in_relative and not to_relative:
                    b1, b2, b3, b4 = box
                    box = [b1 * img_w, b2 * img_h, b3 * img_w, b4 * img_h]
            else:
                value_error_log(logger,
                                "to_source, shape_source, and shape should contain values and should not be None")

        box = Box.get_type(box, out_type)
        if return_int:
            box = [int(b) for b in box]

        return box

    @staticmethod
    def get_type(in_, out_type):
        if out_type is not None:
            try:
                in_ = out_type(in_)
            except:
                raise Exception(
                    f"{out_type} is not Supported. Supported types: {Box._get_enum_names(Box.OutType)}"
                )
        return in_

    @staticmethod
    def get_enum_names(in_):
        return [n.name for n in in_]

    @staticmethod
    def _put_box_pil(img,
                     box,
                     outline="green",
                     fill=None,
                     in_relative=False,
                     in_format="XYXY",
                     in_source="Numpy",
                     return_numpy=True):
        from PIL import ImageDraw, Image
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        box = Box.box2box(
            box,
            in_format=in_format,
            to_format=Box.BoxFormat.XYXY,
            in_source=in_source,
            to_source=Box.BoxSource.Numpy,
            in_relative=in_relative,
            to_relative=False,
            shape=img.size,
            shape_source="Numpy",
        )

        shape = ((box[0], box[1]), (box[2], box[3]))
        draw = ImageDraw.Draw(img)
        draw.rectangle(shape, fill=fill, outline=outline)
        if return_numpy:
            np.array(img)
        return img

    @staticmethod
    def put_box_pil(
            img,
            box,
            outline="green",
            fill=None,
            in_relative=False,
            in_format="XYXY",
            in_source="Numpy",
            return_numpy=True
    ):
        if box is None or len(box) == 0:
            pass
        elif isinstance(box[0], (tuple, list, np.ndarray)):
            for b in box:
                img = Box._put_box_pil(
                    img=img,
                    box=b,
                    outline=outline,
                    fill=fill,
                    in_relative=in_relative,
                    in_format=in_format,
                    in_source=in_source,
                    return_numpy=return_numpy)
        else:
            img = Box._put_box_pil(
                img=img,
                box=box,
                outline=outline,
                fill=fill,
                in_relative=in_relative,
                in_format=in_format,
                in_source=in_source,
                return_numpy=return_numpy)

        return img

    @staticmethod
    def _put_box(
            img,
            box,
            copy=False,
            color=(0, 255, 0),
            thickness=1,
            lineType=None,
            shift=None,
            in_relative=False,
            in_format="XYXY",
            in_source="Numpy",
    ):

        import cv2

        box = Box.box2box(
            box,
            in_format=in_format,
            to_format=Box.BoxFormat.XYXY,
            in_source=in_source,
            to_source=Box.BoxSource.CV,
            in_relative=in_relative,
            to_relative=False,
            shape=img.shape[:2],
            shape_source="Numpy",
        )

        if not isinstance(img, np.ndarray):
            img = np.array(img).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        box = [int(point) for point in box]
        if copy:
            img = img.copy()
        img = cv2.rectangle(
            img,
            (box[0], box[1]),
            (box[2], box[3]),
            color=color,
            thickness=thickness,
            lineType=lineType,
            shift=shift,
        )
        return img

    @staticmethod
    def put_box(
            img,
            box,
            copy=False,
            color=(0, 255, 0),
            thickness=1,
            lineType=None,
            shift=None,
            in_relative=False,
            in_format=BoxFormat.XYXY,
            in_source=BoxSource.Numpy,
    ):
        if box is None or len(box) == 0:
            pass
        elif isinstance(box[0], (tuple, list, np.ndarray)):
            for b in box:
                img = Box._put_box(
                    img,
                    box=b,
                    copy=copy,
                    color=color,
                    thickness=thickness,
                    lineType=lineType,
                    shift=shift,
                    in_format=in_format,
                    in_source=in_source,
                    in_relative=in_relative,
                )
        else:
            img = Box._put_box(
                img,
                box=box,
                copy=copy,
                color=color,
                thickness=thickness,
                lineType=lineType,
                shift=shift,
                in_format=in_format,
                in_source=in_source,
                in_relative=in_relative,
            )

        return img

    @staticmethod
    def _get_box_img(img, bbox, box_format=BoxFormat.XYXY, box_source=BoxSource.Numpy):
        bbox = Box.box2box(
            bbox,
            in_format=box_format,
            to_format=Box.BoxFormat.XYXY,
            in_source=box_source,
            to_source=Box.BoxSource.Numpy,
            return_int=True,
        )
        img_part = img[bbox[0]: bbox[2], bbox[1]: bbox[3]]
        return img_part

    @staticmethod
    def get_box_img(img, bbox, box_format=BoxFormat.XYXY, box_source=BoxSource.Numpy):
        if len(img.shape) != 3:
            raise Exception("The image size should be 3")

        img_part = []
        if bbox is None or len(bbox) == 0:
            pass
        elif isinstance(bbox[0], (tuple, list, np.ndarray)):
            img_part = [Box._get_box_img(
                img, b, box_format, box_source) for b in bbox]
        else:
            img_part = Box._get_box_img(img, bbox, box_format, box_source)
        return img_part

    @staticmethod
    def _put_text_pil(img, text, org, color=(0, 255, 0), font=None, font_size=32, return_np=True):
        from PIL import Image, ImageFont, ImageDraw

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if isinstance(font, str):
            font = ImageFont.truetype(font, font_size)

        draw = ImageDraw.Draw(img)
        draw.text(org, text, color, font=font)
        if return_np:
            img = np.array(img)
        return img

    @staticmethod
    def put_text_pil(
            img,
            text,
            org,
            color=(0, 255, 0),
            font=None,
            font_size: int = 32,
            return_np=True
    ):
        if text is None or len(text) == 0 or org is None or len(org) == 0:
            pass
        elif isinstance(text, (tuple, list, np.ndarray)):
            for t, o in zip(text, org):
                img = Box._put_text_pil(img, t, o, color, font, font_size, return_np)
        else:
            img = Box._put_text_pil(img, text, org, color, font, font_size, return_np)
        return img

    @staticmethod
    def _put_text(
            img,
            text,
            org,
            fontFace=None,
            fontScale=1,
            color=(0, 255, 0),
            thickness=1,
            lineType=None,
            bottomLeftOrigin=None,
            org_source="Numpy"
    ):
        import cv2

        org = (int(org[0]), int(org[1]))
        org = Point.point2point(
            org, in_source=org_source, to_source=Point.PointSource.CV
        )
        font_face = cv2.FONT_HERSHEY_PLAIN if fontFace is None else fontFace
        img = cv2.putText(
            img,
            text,
            org,
            font_face,
            fontScale,
            color,
            thickness,
            lineType,
            bottomLeftOrigin,
        )

        return img

    @staticmethod
    def put_text(
            img,
            text,
            org,
            fontFace=None,
            fontScale: float = 1,
            color=(0, 255, 0),
            thickness=1,
            lineType=None,
            bottomLeftOrigin=None,
            org_source="Numpy",
    ):
        if text is None or len(text) == 0 or org is None or len(org) == 0:
            pass
        elif isinstance(text, (tuple, list, np.ndarray)):
            for t, o in zip(text, org):
                img = Box._put_text(
                    img,
                    t,
                    o,
                    fontFace,
                    fontScale,
                    color,
                    thickness,
                    lineType,
                    bottomLeftOrigin,
                    org_source=org_source,
                )
        else:
            img = Box._put_text(
                img,
                text,
                org,
                fontFace,
                fontScale,
                color,
                thickness,
                lineType,
                bottomLeftOrigin,
                org_source=org_source,
            )
        return img

    @staticmethod
    def get_biggest(
            box,
            in_format=BoxFormat.XYXY,
            in_source=BoxSource.Numpy,
            get_index=False,
            inputs: Union[None, dict] = None,
            reverse=False,
    ):
        if len(box) == 0 or box is None:
            return
        box = Box.box2box(
            box,
            in_format=in_format,
            in_source=in_source,
            to_source=Box.BoxSource.Numpy,
            to_format=Box.BoxFormat.XYWH,
        )
        if reverse:
            chosen_box = min(box, key=lambda b: b[2] * b[3])
        else:
            chosen_box = max(box, key=lambda b: b[2] * b[3])
        index = box.index(chosen_box)
        if inputs is not None:
            inputs = {k: v[index] for k, v in inputs.items()}
            return inputs
        chosen_box = Box.box2box(
            chosen_box, in_format=Box.BoxFormat.XYWH, to_format=Box.BoxFormat.XYXY
        )
        if get_index:
            return chosen_box, index
        return chosen_box

    @staticmethod
    def get_area(box, in_format=BoxFormat.XYXY, in_source=BoxSource.Numpy):
        box = Box.box2box(
            box,
            in_format=in_format,
            in_source=in_source,
            to_source=Box.BoxSource.Numpy,
            to_format=Box.BoxFormat.XYWH,
        )
        area = box[2] * box[3]
        return area

    @staticmethod
    def fill_box(img, box, value, in_format=BoxFormat.XYXY, in_source=BoxSource.Numpy):
        """
        Fill the selected box with the specified value.
        :param img: The input image
        :param box: the box that should be filled
        :param value: the value with which the box will be filled
        :param in_format: box input format
        :param in_source: box input source
        :return: the filled box
        """
        bbox = Box.box2box(
            box,
            in_format=in_format,
            in_source=in_source,
            to_source=Box.BoxSource.Numpy,
            to_format=Box.BoxFormat.XYXY,
        )
        img[bbox[0]: bbox[2], bbox[1]: bbox[3]] = value
        return img

    @staticmethod
    def fill_outer_box(
            img, box, value: int = 0, in_format=BoxFormat.XYXY, in_source=BoxSource.Numpy
    ):
        """
        Fill the outer area of the selected box with the specified value.
        :param img: The input image
        :param box: the box that should remain fixed
        :param value: the value with which the outer box will be filled, default is zero
        :param in_format: box input format
        :param in_source: box input source
        :return: the filled box
        """
        import cv2

        bbox = Box.box2box(
            box,
            in_format=in_format,
            in_source=in_source,
            to_source=Box.BoxSource.Numpy,
            to_format=Box.BoxFormat.XYXY,
        )
        mask = np.ones_like(img, dtype=np.uint8) * value
        mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] = 1
        img = cv2.multiply(img, mask)
        return img

    @staticmethod
    def _put_box_text_pil(img, box, label, outline, text_color, text_font_size, text_font, return_numpy=True):
        from PIL import ImageDraw, Image, ImageFont
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
        else:
            pil_img = img
        if text_font is None:
            pil_font = ImageFont.load_default()
        elif isinstance(text_font, str):
            pil_font = ImageFont.truetype(text_font, text_font_size)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        pil_img = Box.put_box_pil(pil_img, [*p1, *p2], outline=outline, return_numpy=return_numpy)

        draw_interface = ImageDraw.Draw(pil_img)
        w, h = draw_interface.textsize(label, pil_font)
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, (p1[1] - h - 3) if outside else (p1[1] + h + 3)
        pil_img = Box.put_box_pil(pil_img, [*p1, *p2], outline=outline, fill=outline, return_numpy=return_numpy)

        org = p1[0], (p1[1] - h - 2) if outside else (p1[1] + h + 2)

        pil_img = Box.put_text_pil(pil_img, label, org, color=text_color, font_size=text_font_size, font=pil_font,
                                   return_np=return_numpy)

        img = np.array(pil_img) if return_numpy else pil_img
        return img

    @staticmethod
    def put_box_text_pil(
            img: Union[Sequence, np.ndarray],
            box: Union[Sequence],
            label: Union[Sequence, str],
            outline=(128, 128, 128),
            txt_color=(255, 255, 255),
            text_font_size=32,
            text_font=None,
            return_numpy=True,
            in_source=BoxSource.CV,
    ):
        box = Box.box2box(
            box, in_source=in_source, to_source=Box.BoxSource.Numpy
        )
        if (
                isinstance(box, Sequence)
                and isinstance(box[0], Sequence)
                and isinstance(label, Sequence)
        ):
            if isinstance(outline, Sequence) and isinstance(outline[0], Sequence):
                if isinstance(txt_color, Sequence) and isinstance(txt_color[0], Sequence):
                    for b, l, c, t_c in zip(box, label, outline, txt_color):
                        img = Box._put_box_text_pil(img, b, l, c, t_c, text_font_size, text_font, return_numpy)
                else:
                    for b, l, c in zip(box, label, outline):
                        img = Box._put_box_text_pil(img, b, l, c, txt_color, text_font_size, text_font,
                                                    return_numpy)
            else:
                for b, l in zip(box, label):
                    img = Box._put_box_text_pil(img, b, l, outline, txt_color, text_font_size, text_font,
                                                return_numpy)
        else:
            img = Box._put_box_text_pil(img, box, label, outline, txt_color, text_font_size, text_font,
                                        return_numpy)
        return img

    @staticmethod
    def _put_box_text(
            img, box, label, color=(128, 128, 128), txt_color=(255, 255, 255), thickness=2
    ):
        import cv2

        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        img = Box.put_box(
            img,
            box,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
            in_source=Box.BoxSource.CV,
        )
        text_font = max(thickness - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=thickness / 3, thickness=text_font)[
            0
        ]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        img = Box.put_box(
            img,
            [*p1, *p2],
            color=color,
            thickness=-1,
            lineType=cv2.LINE_AA,
            in_source=Box.BoxSource.CV,
        )
        x0, x1 = p1[0], p1[1] - 2 if outside else p1[1] + h + 2
        img = Box.put_text(
            img,
            label,
            (x0, x1),
            fontFace=0,
            fontScale=thickness / 3,
            color=txt_color,
            thickness=text_font,
            lineType=cv2.LINE_AA,
            org_source="CV",
        )
        return img

    @staticmethod
    def put_box_text(
            img: Union[Sequence, np.ndarray],
            box: Union[Sequence],
            label: Union[Sequence, str],
            color=(128, 128, 128),
            txt_color=(255, 255, 255),
            thickness=2,
    ):
        """
        :param img:
        :param box: It should be in numpy source!
        :param label:
        :param color:
        :param txt_color:
        :param thickness:
        :return:
        """
        box = Box.box2box(
            box, in_source=Box.BoxSource.Numpy, to_source=Box.BoxSource.CV
        )
        if (
                isinstance(box, Sequence)
                and isinstance(box[0], Sequence)
                and isinstance(label, Sequence)
        ):
            if isinstance(color, Sequence) and isinstance(color[0], Sequence):
                if isinstance(txt_color, Sequence) and isinstance(
                        txt_color[0], Sequence
                ):
                    for b, l, c, t_c in zip(box, label, color, txt_color):
                        img = Box._put_box_text(img, b, l, c, t_c, thickness)
                else:
                    for b, l, c in zip(box, label, color):
                        img = Box._put_box_text(
                            img, b, l, c, txt_color, thickness)
            else:
                for b, l in zip(box, label):
                    img = Box._put_box_text(
                        img, b, l, color, txt_color, thickness)
        else:
            img = Box._put_box_text(
                img, box, label, color, txt_color, thickness)
        return img

    @classmethod
    def _get_enum_names(cls, PointSource):
        pass
