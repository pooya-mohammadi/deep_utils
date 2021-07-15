import numpy as np
from enum import Enum


class Box:
    class BoxFormat(Enum):
        XYWC = "XYWC"
        XYXY = "XYXY"

    class BoxSource(Enum):
        CV = 'CV'
        Numpy = 'Numpy'
        Torch = 'Torch'
        TF = "TF"

    class OutType(Enum):
        Numpy = np.array
        List = list
        Tuple = tuple

    @staticmethod
    def box2box(box,
                in_format=None,
                to_format=None,
                in_source=None,
                to_source=None,
                relative=None,
                img_w=None,
                img_h=None,
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
        if type(in_format) is Box.BoxFormat:
            in_format = in_format.value
        if type(to_format) is Box.BoxFormat:
            to_format = to_format.value

        if type(in_source) is Box.BoxSource:
            in_source = in_source.value
        if type(to_source) is Box.BoxSource:
            to_source = to_source.value

        if in_format == Box.BoxFormat.XYWC.value and to_format == Box.BoxFormat.XYXY.value:
            x1, y1, w, h = box
            x2, y2 = x1 + w, y1 + h
            box = [x1, y1, x2, y2]
        elif in_format == Box.BoxFormat.XYXY.value and to_format == Box.BoxFormat.XYWC.value:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            box = [x1, y1, w, h]
        elif (in_format is None and to_format is None) or in_format == to_format:
            pass
        else:
            raise Exception(
                f'Conversion form {in_format} to {to_format} is not Supported.'
                f' Supported types: {Box.__get_enum_names(Box.BoxFormat)}')

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
                f' Supported types: {Box.__get_enum_names(Box.BoxSource)}')
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
                    f'{out_type} is not Supported. Supported types: {Box.__get_enum_names(Box.OutType)}')
        return in_

    @staticmethod
    def __get_enum_names(in_):
        return [n.name for n in in_]

    @staticmethod
    def _put_box(img, box, copy=False,
                 color=(0, 255, 0),
                 thickness=1,
                 lineType=None,
                 shift=None,
                 in_format="XYXY",
                 in_source='Numpy'):
        import cv2
        box = Box.box2box(box, in_format=in_format, to_format=Box.BoxFormat.XYXY,
                          in_source=in_source, to_source=Box.BoxSource.CV)

        if type(img) is not np.ndarray:
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
                in_format=BoxFormat.XYXY,
                in_source=BoxSource.Numpy):
        if box is None and len(box) == 0:
            pass
        elif type(box[0]) in [tuple, list, np.ndarray]:
            for b in box:
                img = Box._put_box(img, b, copy, color, thickness, lineType, shift, in_format, in_source)
        else:
            img = Box._put_box(img, box, copy, color, thickness, lineType, shift, in_format, in_source)

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
        if bbox is None and len(bbox) == 0:
            pass
        elif type(bbox[0]) in [tuple, list, np.ndarray]:
            img_part = [Box._get_box_img(img, b, box_format, box_source) for b in bbox]
        else:
            img_part = Box._get_box_img(img, bbox, box_format, box_source)
        return img_part

    @staticmethod
    def _put_text(img, text, org, fontFace=None, fontScale=1, color=(0, 255, 0),
                  thickness=1,
                  lineType=None,
                  bottomLeftOrigin=None):
        import cv2
        font_face = cv2.FONT_HERSHEY_PLAIN if fontFace is None else fontFace
        img = cv2.putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)

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
                 bottomLeftOrigin=None):
        if text is None and len(text) == 0 and org is  None and len(org) ==0:
            pass
        elif type(text[0]) in [tuple, list, np.ndarray]:
            for t, o in zip(text, org):
                img = Box._put_text(img, t, o, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)
        else:
            img = Box._put_text(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)
        return img


if __name__ == '__main__':
    print(Box.BoxFormat.XYXY is Box.BoxFormat)
