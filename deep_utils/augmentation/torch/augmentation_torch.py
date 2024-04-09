from deep_utils.utils.opencv_utils.opencv_utils import CVUtils


class AugmentTorch:
    @staticmethod
    def random_resized_crop(
            size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation="BILINEAR"
    ):
        from torchvision import transforms

        if interpolation == "BILINEAR":
            from torchvision.transforms import InterpolationMode

            interpolation = InterpolationMode.BILINEAR
        return transforms.RandomResizedCrop(
            size, scale=scale, ratio=ratio, interpolation=interpolation
        )

    @staticmethod
    def five_crop(size):
        from torchvision import transforms

        return transforms.FiveCrop(size)

    @staticmethod
    def ten_crop(size):
        from torchvision import transforms

        return transforms.TenCrop(size)

    @staticmethod
    def random_horizontal_flip(p=0.5):
        from torchvision import transforms

        return transforms.RandomHorizontalFlip(p=p)

    @staticmethod
    def random_affine(
            degrees,
            translate=None,
            scale=None,
            shear=None,
            interpolation="NEAREST",
            fill=0,
            fillcolor=None,
            resample=None,
    ):
        from torchvision import transforms

        if interpolation == "NEAREST":
            from torchvision.transforms import InterpolationMode

            interpolation = InterpolationMode.NEAREST
        return transforms.RandomAffine(
            degrees,
            translate=translate,
            scale=scale,
            shear=shear,
            interpolation=interpolation,
            fill=fill,
            fillcolor=fillcolor,
            resample=resample,
        )

    @staticmethod
    def random_rotation(
            degrees,
            interpolation="NEAREST",
            expand=False,
            center=None,
            fill=0,
            resample=None,
    ):
        from torchvision import transforms

        if interpolation == "NEAREST":
            from torchvision.transforms import InterpolationMode

            interpolation = InterpolationMode.NEAREST
        return transforms.RandomRotation(
            degrees,
            interpolation=interpolation,
            expand=expand,
            center=center,
            fill=fill,
            resample=resample,
        )

    @staticmethod
    def gaussian_blur(kernel_size, sigma=(0.1, 2.0)):
        from torchvision import transforms

        return transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    @staticmethod
    def random_erasing(
            p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
    ):
        from torchvision import transforms

        return transforms.RandomErasing(
            p=p, scale=scale, ratio=ratio, value=value, inplace=inplace
        )

    @staticmethod
    def center_crop(size):
        from torchvision import transforms

        return transforms.CenterCrop(size)

    @staticmethod
    def color_jitter(brightness=0, contrast=0, saturation=0, hue=0):
        from torchvision import transforms

        return transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )

    @staticmethod
    def random_vertical_flip(p=0.5):
        from torchvision import transforms

        return transforms.RandomVerticalFlip(p=p)

    @staticmethod
    def pad(padding, fill=0, padding_mode="constant"):
        from torchvision import transforms

        return transforms.Pad(padding=padding, fill=fill, padding_mode=padding_mode)

    @staticmethod
    def random_crop(
            size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"
    ):
        from torchvision import transforms

        return transforms.RandomCrop(
            size=size,
            padding=padding,
            pad_if_needed=pad_if_needed,
            fill=fill,
            padding_mode=padding_mode,
        )

    @staticmethod
    def random_perspective(
            distortion_scale=0.5, p=0.5, interpolation="BILINEAR", fill=0
    ):
        from torchvision import transforms

        if interpolation == "BILINEAR":
            from torchvision.transforms import InterpolationMode

            interpolation = InterpolationMode.BILINEAR
        return transforms.RandomPerspective(
            distortion_scale=distortion_scale,
            p=p,
            interpolation=interpolation,
            fill=fill,
        )

    @staticmethod
    def resize(size, interpolation="BILINEAR"):
        from torchvision import transforms

        if interpolation == "BILINEAR":
            from torchvision.transforms import InterpolationMode

            interpolation = InterpolationMode.BILINEAR
        return transforms.Resize(size=size, interpolation=interpolation)

    @staticmethod
    def normalize(mean, std, inplace=False):
        from torchvision import transforms

        return transforms.Normalize(mean=mean, std=std, inplace=inplace)

    @staticmethod
    def to_tensor():
        from torchvision import transforms

        return transforms.ToTensor()

    @staticmethod
    def get_augments(*args):
        import torch
        from torchvision import transforms

        transformations = []
        use_lambda = False
        for arg in args:
            prob = 0
            arg_type = type(arg)
            if arg_type is tuple:
                arg, prob = arg
            if type(arg) is callable:
                transform = arg()
            else:
                transform = arg
            if prob:
                transform = transforms.RandomApply([transform], p=prob)
            if isinstance(transform, (transforms.TenCrop, transforms.FiveCrop)):
                use_lambda = True
            if isinstance(transform, transforms.Normalize):
                if use_lambda:
                    to_append = transforms.Lambda(
                        lambda tensors: torch.stack(
                            [AugmentTorch.to_tensor()(t) for t in tensors]
                        )
                    )
                else:
                    to_append = AugmentTorch.to_tensor()
                transformations.append(to_append)
            if use_lambda:
                transform = transforms.Lambda(
                    lambda tensors: torch.stack(
                        [transform(t) for t in tensors])
                )
            transformations.append(transform)
        return transforms.Compose(transformations)

    @staticmethod
    def visualize_data_loader(data_loader, mean, std):
        """
        This function is used to convert tensor images to numpy images that will be shown by opencv!
        :param data_loader:
        :param mean:
        :param std:
        :return:
        """
        for x_data, *_ in data_loader:
            for image_tensor in x_data:
                cv2_image = AugmentTorch.tensor_to_image(image_tensor, mean, std)
                CVUtils.show_destroy_cv2(cv2_image)

    @staticmethod
    def tensor_to_image(tensor, mean=None, std=None, return_array=True):
        """
        Note: Whenever the std is 255 the code returns wrong answers because converting to PIL format by default convert the
         image to type uint8 which has 255 in it as well.
        :param tensor:
        :param mean:
        :param std:
        :param return_array:
        :return:
        """
        import numpy as np
        import torch
        from torchvision import transforms

        if mean is not None or std is not None:
            c = tensor.size()[0]
            if std is None:
                std = torch.ones(c)
            elif mean is None:
                mean = torch.zeros(c)

            tensor = transforms.Normalize(
                mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
            )(tensor)

        image = transforms.ToPILImage()(tensor)
        if return_array:
            image = np.array(image)
        return image
