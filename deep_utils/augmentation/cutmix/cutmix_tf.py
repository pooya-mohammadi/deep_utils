import numpy as np

from deep_utils.utils.utils.shuffle_utils import shuffle_group


class CutMixTF:
    @staticmethod
    def _get_bbox(size, lam):
        """
        Get random bbox from input size with the given lambda
        :param size:
        :param lam:
        :return:
        """
        w = size[0]
        h = size[1]

        cut_rat = np.sqrt(1.0 - lam)

        r_w = np.int(w * cut_rat)
        r_h = np.int(h * cut_rat)
        assert np.isclose((r_h * r_w) / (w * h), 1 - lam, atol=0.01)

        # uniform
        r_x = np.random.randint(w)
        r_y = np.random.randint(h)

        x1 = np.clip(r_x - r_w // 2, 0, w)
        y1 = np.clip(r_y - r_h // 2, 0, h)
        x2 = np.clip(r_x + r_w // 2, 0, w)
        y2 = np.clip(r_y + r_h // 2, 0, h)

        return x1, y1, x2, y2

    @staticmethod
    def get_bbox(sizes, lam):
        """
        Get cutmix bboxes from input given sizes
        :param sizes:
        :param lam:
        :return:
        """
        if len(sizes) == 3:
            b = sizes[0]
            boxes = np.array([CutMixTF._get_bbox(sizes[1:], lam)
                             for _ in range(b)])
        elif len(sizes) == 2:
            boxes = np.array(CutMixTF._get_bbox(sizes, lam))
        else:
            raise ValueError(f"input size: {len(sizes)} is not valid!")
        return boxes

    @staticmethod
    def seg_cutmix_batch(
        a_images, a_masks, b_images=None, b_masks=None, beta=1, shuffle=True
    ):
        """
        Cutmix operation for two batch of segmentation images with their corresponding masks! In case of None input
        for `b_images` and `b_masks`, `a_images` and `b_images` are used instead of them.
        :param beta:
        :param a_images:
        :param a_masks:
        :param b_images:
        :param b_masks:
        :param shuffle:
        :return:
        """
        if b_images is None:
            b_images = a_images
            b_masks = a_masks

        a_images, a_masks, b_images, b_masks = (
            a_images.copy(),
            a_masks.copy(),
            b_images.copy(),
            b_masks.copy(),
        )

        if shuffle:
            shuffle_group(a_images, a_masks)
            shuffle_group(b_images, b_masks)

        x_cutmix, y_cutmix = [], []
        for a_img, b_img, a_mask, b_mask in zip(a_images, b_images, a_masks, b_masks):
            x, y = CutMixTF.seg_cutmix(a_img, a_mask, b_img, b_mask, beta)
            x_cutmix.append(x)
            y_cutmix.append(y)
        return np.array(x_cutmix, dtype=np.uint8), np.array(y_cutmix, dtype=np.uint8)

    @staticmethod
    def seg_cutmix(a_img, a_mask, b_img, b_mask, beta=1):
        """
        Cutmix operation for two individual segmentation images with their corresponding masks!
        :param a_img:
        :param a_mask:
        :param b_img:
        :param b_mask:
        :param beta: alpha value for beta distribution, default is one as suggested by the paper!
        :return:
        """
        # Get lambda & box
        lam = np.random.beta(beta, beta)
        (x1, y1, x2, y2) = CutMixTF.get_bbox(a_img.shape[:-1], lam)
        # generate mask
        mask = np.ones_like(a_img)
        mask[x1:x2, y1:y2, :] = 0
        # generate x
        x = (np.multiply(a_img, mask) + np.multiply(b_img, (abs(1.0 - mask)))).astype(
            np.uint8
        )
        # generate y
        mask = mask[:, :, 0]
        y = (np.multiply(a_mask, mask) + np.multiply(b_mask, (abs(1.0 - mask)))).astype(
            np.uint8
        )
        return x, y

    @staticmethod
    def cls_cutmix_batch(
        a_images, a_labels, b_images=None, b_labels=None, beta=1, shuffle=True
    ):
        """

        :param a_images:
        :param a_labels:
        :param b_images:
        :param b_labels:
        :param beta:
        :param input_type:
        :param shuffle:
        :return:
        """

        if b_images is None:
            b_images = a_images
            b_labels = a_labels

        a_images, a_labels, b_images, b_labels = (
            a_images.copy(),
            a_labels.copy(),
            b_images.copy(),
            b_labels.copy(),
        )

        if shuffle:
            shuffle_group(a_images)
            shuffle_group(b_images)

        x_cutmix, y_cutmix = [], []
        for a_img, b_img, a_label, b_label in zip(
            a_images, b_images, a_labels, b_labels
        ):
            img_cutmix, label_cutmix = CutMixTF._cls_cutmix(
                a_images, a_img, a_label, b_images, b_label, beta
            )
            x_cutmix.append(img_cutmix)
            y_cutmix.append(label_cutmix)

        return np.array(x_cutmix), np.array(y_cutmix)

    @staticmethod
    def _cls_cutmix(a_images, a_img, a_label, b_images, b_label, beta):
        lam = np.random.beta(beta, beta)
        (x1, y1, x2, y2) = CutMixTF.get_bbox(a_img.shape, lam)
        # create mask
        img_cutmix_mask = np.ones_like(a_img)
        img_cutmix_mask[x1:x2, y1:y2, :] = 0
        img_cutmix = (
            np.multiply(a_images, img_cutmix_mask)
            + np.multiply(b_images, (abs(1.0 - img_cutmix_mask)))
        ).astype(np.uint8)
        label_cutmix = lam * a_label + b_label * (1 - lam)
        return img_cutmix, label_cutmix
