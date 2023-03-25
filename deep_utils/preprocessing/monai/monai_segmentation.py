from typing import List
import torch
from monai.transforms import MapTransform, ScaleIntensity, ScaleIntensityRange


class MonaiChannelBasedContrastEnhancementD(MapTransform):
    """
    Channel based Enhancement of input channels.
    """

    def __init__(self, keys, in_channels: List[str], contrast_enhancement_params: dict = None):
        """
        The keys input is used by the monai mapping. The in_channels should be based on the order of the channels. For
        instance, if you have stacked your list of image in ["CT", "PET"] order then the in_channels should be equal
        to ["CT", "PET"].
        The other parameter is `contrast_enhancement_params`. This is a dictionary that like the following dict:
         dict(MRI=(ScaleIntensity, dict(minv=0.0, maxv=1.0)),
              CT=(ScaleIntensityRange, dict(a_min=0, a_max=1, b_min=-1, b_max=1, clip=True)),
              PET=(ScaleIntensity, dict(minv=0.0, maxv=1.0)))
        The key is the name of the channel and the value is consisted of two section, the first section is enhancement
         class such as ScaleIntensity and the second section is parameters required for that class.
         Note: you can provide more items in contrast_enhancement_params, however, you have to make sure to provide
         the ones you have listed in in_channels argument.
        :param keys:
        :param in_channels:
        :param contrast_enhancement_params:
        """
        super().__init__(keys=keys)
        self.in_channels = in_channels
        contrast_enhancement_params = contrast_enhancement_params if contrast_enhancement_params is not None else dict(
            MRI=(ScaleIntensity, dict(minv=0.0, maxv=1.0)),
            CT=(ScaleIntensityRange, dict(a_min=0, a_max=1, b_min=-1, b_max=1, clip=True)),
            PET=(ScaleIntensity, dict(minv=0.0, maxv=1.0)),
        )
        for channel_name in self.in_channels:
            if channel_name not in contrast_enhancement_params.keys():
                raise ValueError(f"The channel name: {channel_name} is not available in the provided contrast"
                                 f" enhancement: {contrast_enhancement_params.keys()}")
        self.scalars = {img_name: type_(**params) for img_name, (type_, params) in contrast_enhancement_params.items()}

    def __call__(self, data):
        """
        This is called by monai mapper.
        :param data:
        :return:
        """
        d = dict(data)
        for key in self.keys:
            result = []
            for idx, in_channel in enumerate(self.in_channels):
                if in_channel in self.scalars:
                    result.append(self.scalars[in_channel](d[key][idx]))
                else:
                    result.append(d[key][idx])
            d[key] = torch.stack(result, axis=0).float()
        return d
