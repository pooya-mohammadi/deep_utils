import numpy as np
from PIL import Image
from deep_utils.utils.box_utils.box_dataclasses import BoxDataClass
from deep_utils.utils.box_utils.boxes import Box
import torch
from torch.nn import functional as F
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)
from glide_text2im.download import load_checkpoint
from typing import Union, List
from dataclasses import dataclass
from enum import Enum


class ImageEditingGLIDETypes(str, Enum):
    Fast = "fast"
    Slow = "slow"


@dataclass
class ImageEditingGLIDEOptions:
    img_small_size: int
    img_large_size: int
    diffusion_steps: str
    upsampling_steps: Union[str, int]
    upsample_temp: float
    guidance_scale: float


class ImageEditingGLIDE:
    EXECUTE_DETAILS = {
        ImageEditingGLIDETypes.Fast: ImageEditingGLIDEOptions(
            img_small_size=64,
            img_large_size=256,
            diffusion_steps="100",
            upsampling_steps="fast27",
            upsample_temp=0.997,
            guidance_scale=0.5,
        ),
        ImageEditingGLIDETypes.Slow: ImageEditingGLIDEOptions(
            img_small_size=64,
            img_large_size=256,
            diffusion_steps="500",
            upsampling_steps="100",
            upsample_temp=0.99,
            guidance_scale=0.5,
        )
    }

    def __init__(self,
                 execute_type: ImageEditingGLIDETypes = ImageEditingGLIDETypes.Fast,
                 inpaint_model_path=None,
                 upsampling_model_path=None,
                 device='cuda'
                 ):
        execute_details = self.EXECUTE_DETAILS[execute_type]
        self._img_large_size = execute_details.img_large_size
        self._img_small_size = execute_details.img_small_size
        self._upsample_temp = execute_details.upsample_temp
        self._inpaint_model_path = inpaint_model_path
        self._upsampling_model_path = upsampling_model_path
        self._device = device
        self._has_cuda = device == 'cuda'
        self._diffusion_steps = execute_details.diffusion_steps
        self._upsampling_steps = execute_details.upsampling_steps
        self._guidance_scale = execute_details.guidance_scale
        self._inpaint_options, self._upsampling_options = dict(), dict()
        self._inpaint_model, self._inpaint_diffusion = self._load_inpaint_model()
        self._upsampling_model, self._upsampling_diffusion = self._load_upsampling_model()

    def _load_inpaint_model(self):
        # Create base model.
        self._inpaint_options = model_and_diffusion_defaults()
        self._inpaint_options['inpaint'] = True
        self._inpaint_options['use_fp16'] = self._has_cuda
        self._inpaint_options['timestep_respacing'] = self._diffusion_steps  # use 100 diffusion steps for fast sampling
        self._inpaint_options['image_size'] = self._img_small_size
        model, diffusion = create_model_and_diffusion(**self._inpaint_options)
        model.eval()
        if self._has_cuda:
            model.convert_to_fp16()
        model.to(self._device)
        if self._inpaint_model_path is not None:
            model.load_state_dict(torch.load(self._inpaint_model_path, map_location=torch.device(self._device)))
        else:
            model.load_state_dict(load_checkpoint('base-inpaint', torch.device(self._device)))
        print('total base parameters', sum(x.numel() for x in model.parameters()))
        return model, diffusion

    def _load_upsampling_model(self):
        """
        Loads the up-sampling model.
        :return:
        """
        self._upsampling_options = model_and_diffusion_defaults_upsampler()
        self._upsampling_options['inpaint'] = True
        self._upsampling_options['use_fp16'] = self._has_cuda
        self._upsampling_options[
            'timestep_respacing'] = self._upsampling_steps  # use 27 diffusion steps for very fast sampling
        model_up, diffusion_up = create_model_and_diffusion(**self._upsampling_options)
        model_up.eval()
        if self._has_cuda:
            model_up.convert_to_fp16()
        model_up.to(self._device)
        if self._upsampling_model_path is not None:
            model_up.load_state_dict(torch.load(self._upsampling_model_path, map_location=torch.device(self._device)))
        else:
            model_up.load_state_dict(load_checkpoint('upsample-inpaint', torch.device(self._device)))
        print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))
        return model_up, diffusion_up

    @staticmethod
    def _img_preparation(img: Union[np.ndarray, Image.Image], img_size: int) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            img_pil = img
        elif isinstance(img, torch.Tensor):
            img_pil = ImageEditingGLIDE._get_torch_img(img)
        else:
            raise ValueError(f'img should be either np.ndarray or PIL.Image.Image, but got {type(img)}')
        resize_pil_img = img_pil.copy().resize((img_size, img_size), resample=Image.BICUBIC)
        np_img = np.array(resize_pil_img)
        return torch.from_numpy(np_img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1

    def _model_fn(self, x_t, ts, **kwargs) -> torch.Tensor:
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self._inpaint_model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + self._guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    @staticmethod
    def _denoised_fn(model_kwargs: dict):
        def denoised_fn(x_start):
            # Force the model to have the exact right x_start predictions
            # for the part of the image which is known.
            return (
                    x_start * (1 - model_kwargs['inpaint_mask'])
                    + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
            )

        return denoised_fn

    def edit_small_box(self, img: Union[np.ndarray, Image.Image], text: str, box: BoxDataClass):
        if isinstance(box, list):
            box = BoxDataClass.from_list(box)
        batch_size = 1
        source_img_small = self._img_preparation(img, img_size=self._img_small_size)
        resized_box_small = Box.resize_box(box.to_list(), img.size, (self._img_small_size, self._img_small_size))
        resized_box_small = BoxDataClass.from_list(resized_box_small)
        source_mask_small = torch.ones_like(source_img_small)[:, :1]
        source_mask_small[:, :, resized_box_small.p1.x:resized_box_small.p2.x,
        resized_box_small.p1.y:resized_box_small.p2.y] = 0

        tokens = self._inpaint_model.tokenizer.encode(text)
        tokens, mask = self._inpaint_model.tokenizer.padded_tokens_and_mask(
            tokens, self._inpaint_options['text_ctx']
        )

        # Create the classifier-free guidance tokens (empty)
        full_batch_size = batch_size * 2
        uncond_tokens, uncond_mask = self._inpaint_model.tokenizer.padded_tokens_and_mask(
            [], self._inpaint_options['text_ctx']
        )

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=torch.tensor(
                [tokens] * batch_size + [uncond_tokens] * batch_size, device=self._device
            ),
            mask=torch.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=torch.bool,
                device=self._device,
            ),

            # Masked inpainting image
            inpaint_image=(source_img_small * source_mask_small).repeat(full_batch_size, 1, 1, 1).to(self._device),
            inpaint_mask=source_mask_small.repeat(full_batch_size, 1, 1, 1).to(self._device),
        )

        # Create a classifier-free guidance sampling function

        # Sample from the base model.
        self._inpaint_model.del_cache()
        output_images = self._inpaint_diffusion.p_sample_loop(
            self._model_fn,
            (full_batch_size, 3, self._inpaint_options["image_size"],
             self._inpaint_options["image_size"]),
            device=self._device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            denoised_fn=self._denoised_fn(model_kwargs),
        )[:batch_size]
        self._inpaint_model.del_cache()
        return output_images

    @staticmethod
    def denoised_fn(x_start, model_kwargs):
        # Force the model to have the exact right x_start predictions
        # for the part of the image which is known.
        return (
                x_start * (1 - model_kwargs['inpaint_mask'])
                + model_kwargs['inpaint_image'] * model_kwargs['inpaint_mask']
        )

    def upscale(self, edited_img: torch.Tensor, img: Union[np.ndarray, Image.Image], text: str, box: BoxDataClass):
        if isinstance(box, list):
            box = BoxDataClass.from_list(box)

        batch_size = 1
        source_img_small = self._img_preparation(img, img_size=self._img_small_size)
        resized_box_small = Box.resize_box(box.to_list(), img.size, (self._img_small_size, self._img_small_size))
        resized_box_small = BoxDataClass.from_list(resized_box_small)
        source_mask_small = torch.ones_like(source_img_small)[:, :1]
        source_mask_small[:, :, resized_box_small.p1.x:resized_box_small.p2.x,
        resized_box_small.p1.y:resized_box_small.p2.y] = 0
        source_img_large = self._img_preparation(img, img_size=self._img_large_size)
        source_mask_large = F.interpolate(source_mask_small,
                                          (self._img_large_size,
                                           self._img_large_size),
                                          mode='nearest')

        tokens = self._upsampling_model.tokenizer.encode(text)
        tokens, mask = self._upsampling_model.tokenizer.padded_tokens_and_mask(
            tokens, self._upsampling_options['text_ctx']
        )

        # Create the model conditioning dict.
        model_kwargs = dict(
            # Low-res image to upsample.
            low_res=((edited_img + 1) * 127.5).round() / 127.5 - 1,

            # Text tokens
            tokens=torch.tensor(
                [tokens] * batch_size, device=self._device
            ),
            mask=torch.tensor(
                [mask] * batch_size,
                dtype=torch.bool,
                device=self._device,
            ),

            # Masked inpainting image.
            inpaint_image=(source_img_large * source_mask_large).repeat(batch_size, 1, 1, 1).to(self._device),
            inpaint_mask=source_mask_large.repeat(batch_size, 1, 1, 1).to(self._device),
        )

        # Sample from the base model.
        self._upsampling_model.del_cache()
        up_shape = (batch_size, 3, self._upsampling_options["image_size"], self._upsampling_options["image_size"])
        up_samples = self._upsampling_diffusion.p_sample_loop(
            self._upsampling_model,
            up_shape,
            noise=torch.randn(up_shape, device=self._device) * self._upsample_temp,
            device=self._device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
            denoised_fn=self._denoised_fn(model_kwargs),
        )[:batch_size]
        self._upsampling_model.del_cache()

        return up_samples

    def edit_box(self, img: Union[np.ndarray, Image.Image], text: str,
                 box: Union[BoxDataClass, List[float]]) -> Image.Image:
        if isinstance(box, list):
            box = BoxDataClass.from_list(box)

        edited_img = self.edit_small_box(img, text, box)
        upscaled_images = self.upscale(edited_img, img, text, box)
        images = self._get_torch_img(upscaled_images)
        return images

    @staticmethod
    def _get_torch_img(img: torch.Tensor) -> Image.Image:
        """ Display a batch of images inline. """
        scaled = ((img + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu()
        reshaped = scaled.permute(2, 0, 3, 1).reshape([img.shape[2], -1, 3])
        return Image.fromarray(reshaped.numpy())


if __name__ == '__main__':
    from PIL import Image
    from deep_utils import DownloadUtils

    image_download_path = "https://github.com/pooya-mohammadi/deep_utils/releases/download/1.0.2/golsa_in_garden.jpg"
    DownloadUtils.download_file(image_download_path, exists_skip=True)

    editing_model = ImageEditingGLIDE()
    image_path = "golsa_in_garden.jpg"
    box = [340.6672668457031, 403.7683410644531, 372.0812072753906, 439.3288879394531]
    pil_img = Image.open(image_path)
    edit_text = "dead gray leaves"
    output_img = editing_model.edit_box(img=pil_img, text=edit_text, box=box)
    print(output_img.size)
    output_img.save("output.jpg")
