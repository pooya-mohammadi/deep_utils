import math
from collections import defaultdict

import numpy as np

from deep_utils.utils.resize_utils.main_resize import resize

from .box_utils import _preprocess, nms


def run_first_stage(image, net, scale, threshold, device):
    """Run P-Net, generate bounding boxes, and do NMS.
    from deep_utils.utils

        Arguments:
            image: an instance of PIL.Image.
            net: an instance of pytorch's nn.Module, P-Net.
            scale: a float number,
                scale width and height of the image by this number.
            threshold: a float number,
                threshold on the probability of a face when generating
                bounding boxes from predictions of the net.

        Returns:
            a float numpy array of shape [n_boxes, 9],
                bounding boxes with scores and offsets (4 + 1 + 4).
    """

    # scale the image and convert it to a float array
    width, height = image.shape[1:3]
    sw, sh = math.ceil(width * scale), math.ceil(height * scale)
    img = resize(image, (sw, sh))
    img = img.transpose((0, 2, 1, 3))
    img = (img - 127.5) * 0.0078125
    output = net.predict(img)
    out0 = output[0].transpose((0, 3, 2, 1))
    out1 = output[1].transpose((0, 3, 2, 1))
    bboxes = []
    for img_n in range(img.shape[0]):
        probs = out0[img_n, 1, :, :]
        offsets = out1[img_n]
        # probs: probability of a face at each sliding window
        # offsets: transformations to true bounding boxes

        boxes = _generate_bboxes(probs, offsets, scale, threshold)
        if len(boxes) == 0:
            return None

        keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
        bboxes.append(boxes[keep])
    return bboxes


def _generate_bboxes(probs, offsets, scale, threshold):
    """Generate bounding boxes at places
    where there is probably a face.

    Arguments:
        probs: a float numpy array of shape [n, m].
        offsets: a float numpy array of shape [1, 4, n, m].
        scale: a float number,
            width and height of the image were scaled by this number.
        threshold: a float number.

    Returns:
        a float numpy array of shape [n_boxes, 9]
    """
    # applying P-Net is equivalent, in some sense, to
    # moving 12x12 window with stride 2
    stride = 2
    cell_size = 12

    # indices of boxes where there is probably a face
    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return np.array([])

    # transformations of bounding boxes
    tx1, ty1, tx2, ty2 = [offsets[i, inds[0], inds[1]] for i in range(4)]
    # they are defined as:
    # w = x2 - x1 + 1
    # h = y2 - y1 + 1
    # x1_true = x1 + tx1*w
    # x2_true = x2 + tx2*w
    # y1_true = y1 + ty1*h
    # y2_true = y2 + ty2*h

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back
    bounding_boxes = np.vstack(
        [
            np.round((stride * inds[1] + 1.0) / scale),
            np.round((stride * inds[0] + 1.0) / scale),
            np.round((stride * inds[1] + 1.0 + cell_size) / scale),
            np.round((stride * inds[0] + 1.0 + cell_size) / scale),
            score,
            offsets,
        ]
    )
    return bounding_boxes.T
