from typing import Union
import numpy as np
from enum import Enum


class Point:
    class PointSource(Enum):
        Torch = 'Torch'
        TF = "TF"
        CV = 'CV'
        Numpy = 'Numpy'

    @staticmethod
    def point2point(point, in_source, to_source, in_relative=None, to_relative=None, shape=None, shape_source=None):
        if point is None or len(point) == 0:
            pass
        elif isinstance(point[0], (tuple, list, np.ndarray)):
            point = [Point._point2point(p, in_source=in_source, to_source=to_source,
                                        in_relative=in_relative, to_relative=to_relative,
                                        shape=shape, shape_source=shape_source) for p in point]
        else:
            point = Point._point2point(point, in_source=in_source, to_source=to_source,
                                       in_relative=in_relative, to_relative=to_relative,
                                       shape=shape, shape_source=shape_source)
        return point

    @staticmethod
    def _point2point(point, in_source, to_source, in_relative=None, to_relative=None, shape=None, shape_source=None):
        if isinstance(in_source, Point.PointSource):
            in_source = in_source.value
        if isinstance(to_source, Point.PointSource):
            to_source = to_source.value

        if (in_source in [Point.PointSource.Torch.value, Point.PointSource.CV.value] and to_source in [
            Point.PointSource.TF.value, Point.PointSource.Numpy.value]) \
                or (in_source in [Point.PointSource.TF.value, Point.PointSource.Numpy.value] and to_source in [
            Point.PointSource.Torch.value, Point.PointSource.CV.value]):
            point = (point[1], point[0])
        elif (in_source is None and to_source is None) or in_source == to_source \
                or (in_source in [Point.PointSource.Torch.value, Point.PointSource.CV.value] and to_source in [
            Point.PointSource.CV.value, Point.PointSource.Torch.value]) \
                or (in_source in [Point.PointSource.TF.value, Point.PointSource.Numpy.value] and to_source in [
            Point.PointSource.TF.value, Point.PointSource.Numpy.value]):
            pass
        else:
            raise Exception(
                f'Conversion form {in_source} to {to_source} is not Supported.'
                f' Supported types: {Box._get_enum_names(Point.PointSource)}')
        if to_source is not None and shape_source is not None and shape is not None:
            img_w, img_h = Point.point2point(shape, in_source=shape_source, to_source=to_source)
            if not in_relative and to_relative:
                p1, p2 = point
                point = [p1 / img_w, p2 / img_h]
            elif in_relative and not to_relative:
                p1, p2 = point
                point = [p1 * img_w, p2 * img_h]
        return point

    @staticmethod
    def _put_point(img, point, radius, color=(0, 255, 0), thickness=None, lineType=None, shift=None, in_source="Numpy"):
        import cv2
        if not isinstance(point, int):
            point = (int(point[0]), int(point[1]))
        point = Point.point2point(point, in_source=in_source, to_source="CV")
        return cv2.circle(img, point, radius, color, thickness, lineType, shift)

    @staticmethod
    def put_point(img, point, radius, color=(0, 255, 0), thickness=None, lineType=None, shift=None, in_source="Numpy"):
        if point is None or len(point) == 0:
            pass
        elif isinstance(point[0], (tuple, list, np.ndarray)):
            for p in point:
                img = Point._put_point(img, p, radius, color, thickness, lineType, shift, in_source)
        else:
            img = Point._put_point(img, point, radius, color, thickness, lineType, shift, in_source)
        return img


class Box:
    class BoxFormat(Enum):
        XYWH = "XYWH"
        XYXY = "XYXY"
        XCYC = "XCYC"

    class BoxSource(Enum):
        Torch = 'Torch'
        TF = "TF"
        CV = 'CV'
        Numpy = 'Numpy'

    class OutType(Enum):
        Numpy = np.array
        List = list
        Tuple = tuple

    @staticmethod
    def box2box(box,
                in_format=None,
                to_format=None,
                in_source=BoxSource.Numpy,
                to_source=BoxSource.Numpy,
                in_relative=None,
                to_relative=None,
                shape=None,
                shape_source=None,
                out_type=None,
                return_int=None):
        if box is None or len(box) == 0:
            pass
        elif isinstance(box[0], (tuple, list, np.ndarray)):
            box = [Box._box2box(b, in_format=in_format, to_format=to_format, in_source=in_source, to_source=to_source,
                                in_relative=in_relative, to_relative=to_relative, shape=shape,
                                shape_source=shape_source, out_type=out_type, return_int=return_int)
                   for b in box]

        else:
            box = Box._box2box(box, in_format=in_format, to_format=to_format, in_source=in_source, to_source=to_source,
                               in_relative=in_relative, to_relative=to_relative, shape=shape,
                               shape_source=shape_source,
                               out_type=out_type, return_int=return_int)
        return box

    @staticmethod
    def _box2box(box,
                 in_format=None,
                 to_format=None,
                 in_source=None,
                 to_source=None,
                 in_relative=None,
                 to_relative=None,
                 shape=None,
                 shape_source=None,
                 out_type=None,
                 return_int=None):
        """

        :param box:
        :param in_format:
        :param to_format:
        :param in_source:
        :param to_source:
        :param relative:
        :param img_w:
        :param img_h:
        :param out_type: output type of the box. Supported types: list, tuple, numpy
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

        if in_format == Box.BoxFormat.XYWH.value and to_format == Box.BoxFormat.XYXY.value:
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            box = [x1, y1, x2, y2]
        elif in_format == Box.BoxFormat.XYXY.value and to_format == Box.BoxFormat.XYWH.value:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            box = [x1, y1, w, h]
        elif in_format == Box.BoxFormat.XYXY.value and to_format == Box.BoxFormat.XCYC.value:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            box = [xc, yc, w, h]
        elif in_format == Box.BoxFormat.XCYC.value and to_format == Box.BoxFormat.XYXY.value:
            xc, yc, w, h = box
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            box = [x1, y1, x2, y2]
        elif in_format == Box.BoxFormat.XYWH.value and to_format == Box.BoxFormat.XCYC.value:
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            box = [xc, yc, w, h]
        elif in_format == Box.BoxFormat.XCYC.value and to_format == Box.BoxFormat.XYWH.value:
            xc, yc, w, h = box
            x1, y1 = xc - w // 2, yc - h // 2

            box = [x1, y1, w, h]
        elif (in_format is None and to_format is None) or in_format == to_format:
            pass
        else:
            raise Exception(
                f'Conversion form {in_format} to {to_format} is not Supported.'
                f' Supported types: {Box._get_enum_names(Box.BoxFormat)}')

        if (in_source in [Box.BoxSource.Torch.value, Box.BoxSource.CV.value] and to_source in [
            Box.BoxSource.TF.value, Box.BoxSource.Numpy.value]) \
                or (in_source in [Box.BoxSource.TF.value, Box.BoxSource.Numpy.value] and to_source in [
            Box.BoxSource.Torch.value, Box.BoxSource.CV.value]):
            box = [box[1], box[0], box[3], box[2]]
        elif (in_source is None and to_source is None) or in_source == to_source \
                or (in_source in [Box.BoxSource.Torch.value, Box.BoxSource.CV.value] and to_source in [
            Box.BoxSource.CV.value, Box.BoxSource.Torch.value]) \
                or (in_source in [Box.BoxSource.TF.value, Box.BoxSource.Numpy.value] and to_source in [
            Box.BoxSource.TF.value, Box.BoxSource.Numpy.value]):
            pass
        else:
            raise Exception(
                f'Conversion form {in_source} to {to_source} is not Supported.'
                f' Supported types: {Box._get_enum_names(Box.BoxSource)}')
        if to_source is not None and shape_source is not None and shape is not None:
            img_w, img_h = Point.point2point(shape, in_source=shape_source, to_source=to_source)
            if not in_relative and to_relative:
                b1, b2, b3, b4 = box
                box = [b1 / img_w, b2 / img_h, b3 / img_w, b4 / img_h]
            elif in_relative and not to_relative:
                b1, b2, b3, b4 = box
                box = [b1 * img_w, b2 * img_h, b3 * img_w, b4 * img_h]

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
                    f'{out_type} is not Supported. Supported types: {Box._get_enum_names(Box.OutType)}')
        return in_

    @staticmethod
    def _get_enum_names(in_):
        return [n.name for n in in_]

    @staticmethod
    def _put_box(img, box, copy=False,
                 color=(0, 255, 0),
                 thickness=1,
                 lineType=None,
                 shift=None,
                 in_relative=False,
                 in_format="XYXY",
                 in_source='Numpy'):
        import cv2
        box = Box.box2box(box,
                          in_format=in_format,
                          to_format=Box.BoxFormat.XYXY,
                          in_source=in_source,
                          to_source=Box.BoxSource.CV,
                          in_relative=in_relative,
                          to_relative=False,
                          shape=img.shape[:2],
                          shape_source='Numpy')

        if not isinstance(img, np.ndarray):
            img = np.array(img).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        box = [int(point) for point in box]
        if copy:
            img = img.copy()
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=color, thickness=thickness,
                            lineType=lineType,
                            shift=shift)
        return img

    @staticmethod
    def put_box(img, box, copy=False,
                color=(0, 255, 0),
                thickness=1,
                lineType=None,
                shift=None,
                in_relative=False,
                in_format=BoxFormat.XYXY,
                in_source=BoxSource.Numpy):
        if box is None or len(box) == 0:
            pass
        elif isinstance(box[0], (tuple, list, np.ndarray)):
            for b in box:
                img = Box._put_box(img, box=b, copy=copy, color=color, thickness=thickness, lineType=lineType,
                                   shift=shift, in_format=in_format, in_source=in_source, in_relative=in_relative)
        else:
            img = Box._put_box(img, box=box, copy=copy, color=color, thickness=thickness, lineType=lineType,
                               shift=shift, in_format=in_format, in_source=in_source, in_relative=in_relative)

        return img

    @staticmethod
    def _get_box_img(img, bbox, box_format=BoxFormat.XYXY, box_source=BoxSource.Numpy):
        bbox = Box.box2box(bbox, in_format=box_format, to_format=Box.BoxFormat.XYXY, in_source=box_source,
                           to_source=Box.BoxSource.Numpy, return_int=True)
        img_part = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        return img_part

    @staticmethod
    def get_box_img(img, bbox, box_format=BoxFormat.XYXY, box_source=BoxSource.Numpy):
        if len(img.shape) != 3:
            raise Exception('The image size should be 3')

        img_part = []
        if bbox is None or len(bbox) == 0:
            pass
        elif isinstance(bbox[0], (tuple, list, np.ndarray)):
            img_part = [Box._get_box_img(img, b, box_format, box_source) for b in bbox]
        else:
            img_part = Box._get_box_img(img, bbox, box_format, box_source)
        return img_part

    @staticmethod
    def _put_text(img, text, org, fontFace=None, fontScale=1, color=(0, 255, 0),
                  thickness=1, lineType=None, bottomLeftOrigin=None, org_source='Numpy'):
        import cv2
        org = (int(org[0]), int(org[1]))
        org = Point.point2point(org, in_source=org_source, to_source=Point.PointSource.CV)
        font_face = cv2.FONT_HERSHEY_PLAIN if fontFace is None else fontFace
        img = cv2.putText(img, text, org, font_face, fontScale, color, thickness, lineType, bottomLeftOrigin)

        return img

    @staticmethod
    def put_text(img,
                 text,
                 org,
                 fontFace=None,
                 fontScale=1,
                 color=(0, 255, 0),
                 thickness=1,
                 lineType=None,
                 bottomLeftOrigin=None,
                 org_source='Numpy'):
        if text is None or len(text) == 0 or org is None or len(org) == 0:
            pass
        elif isinstance(text, (tuple, list, np.ndarray)):
            for t, o in zip(text, org):
                img = Box._put_text(img, t, o, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin,
                                    org_source=org_source)
        else:
            img = Box._put_text(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin,
                                org_source=org_source)
        return img

    @staticmethod
    def get_biggest(box,
                    in_format=BoxFormat.XYXY,
                    in_source=BoxSource.Numpy,
                    get_index=False,
                    inputs: Union[None, dict] = None,
                    reverse=False):
        if len(box) == 0 or box is None:
            return
        box = Box.box2box(box,
                          in_format=in_format,
                          in_source=in_source,
                          to_source=Box.BoxSource.Numpy,
                          to_format=Box.BoxFormat.XYWH
                          )
        if reverse:
            chosen_box = min(box, key=lambda b: b[2] * b[3])
        else:
            chosen_box = max(box, key=lambda b: b[2] * b[3])
        index = box.index(chosen_box)
        if inputs is not None:
            inputs = {k: v[index] for k, v in inputs.items()}
            return inputs
        chosen_box = Box.box2box(chosen_box, in_format=Box.BoxFormat.XYWH, to_format=Box.BoxFormat.XYXY)
        if get_index:
            return chosen_box, index
        return chosen_box

    @staticmethod
    def get_area(box,
                 in_format=BoxFormat.XYXY,
                 in_source=BoxSource.Numpy):
        box = Box.box2box(box,
                          in_format=in_format,
                          in_source=in_source,
                          to_source=Box.BoxSource.Numpy,
                          to_format=Box.BoxFormat.XYWH
                          )
        area = box[2] * box[3]
        return area

    @staticmethod
    def fill_box(img,
                 box,
                 value,
                 in_format=BoxFormat.XYXY,
                 in_source=BoxSource.Numpy):
        """
        Fill the selected box with the specified value.
        :param img: The input image
        :param box: the box that should be filled
        :param value: the value with which the box will be filled
        :param in_format: box input format
        :param in_source: box input source
        :return: the filled box
        """
        bbox = Box.box2box(box,
                           in_format=in_format,
                           in_source=in_source,
                           to_source=Box.BoxSource.Numpy,
                           to_format=Box.BoxFormat.XYXY
                           )
        img[bbox[0]:bbox[2], bbox[1]:bbox[3]] = value
        return img

    @staticmethod
    def fill_outer_box(img,
                       box,
                       value: int = 0,
                       in_format=BoxFormat.XYXY,
                       in_source=BoxSource.Numpy):
        """
        Fill the outer area of the selected box with the specified value.
        :param img: The input image
        :param box: the box that should remain fixed
        :param value: the value with which the outer box will be filled, default is zero
        :param in_format: box input format
        :param in_source: box input source
        :return: the filled box
        """
        bbox = Box.box2box(box,
                           in_format=in_format,
                           in_source=in_source,
                           to_source=Box.BoxSource.Numpy,
                           to_format=Box.BoxFormat.XYXY
                           )
        mask = np.ones_like(img, dtype=np.uint8) * value
        mask[bbox[0]:bbox[2], bbox[1]:bbox[3]] = 1
        img = cv2.multiply(img, mask)
        return img


if __name__ == '__main__':
    print(Box.BoxFormat.XYXY is Box.BoxFormat)
