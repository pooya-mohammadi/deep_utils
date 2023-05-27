from enum import Enum
from typing import Tuple, List, Union
from deep_utils.utils.box_utils.box_dataclasses import BoxDataClass
from deep_utils.utils.box_utils.boxes import Box
import groundingdino.datasets.transforms as T
import numpy as np
from dataclasses import dataclass
import torch
from PIL import Image
from groundingdino.config import GroundingDINO_SwinT_OGC, GroundingDINO_SwinB_cfg
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap
from torchvision.ops import box_convert
from deep_utils.utils.download_utils.download_utils import DownloadUtils


class Text2BoxVisualGroundingDinoModelTypes(str, Enum):
    SWIN_B = "swin_b"
    SWIN_T = "swin_t"


@dataclass
class Text2BoxVisualGroundingDinoOutput:
    """
    Output of Text2BoxVisualGroundingDinoTorch
    """
    boxes: List[BoxDataClass]
    """
    List of boxes. Each box is a tuple of (p1, p2) where p1 is the top left corner and p2 is the bottom right corner.
    """
    labels: List[str]
    """
    List of text corresponding to each box.
    """
    scores: List[float]

    def __post_init__(self):
        if len(self.boxes) != len(self.labels) or len(self.boxes) != len(self.scores):
            raise ValueError(
                f"boxes, labels and scores must have the same length, but they have {len(self.boxes)}, {len(self.labels)} and {len(self.scores)} respectively.")

    def __str__(self):
        return f"Text2BoxVisualGroundingDinoOutput(boxes={self.boxes}, labels={self.labels}, scores={self.scores})"

    def __repr__(self):
        return self.__str__()


@dataclass
class ModelDetails:
    repo_id: str
    filename: str
    model_url: str


class Text2BoxVisualGroundingDino:
    MODEL_DETAILS = {
        Text2BoxVisualGroundingDinoModelTypes.SWIN_T: ModelDetails(repo_id="ShilongLiu/GroundingDINO",
                                                                   filename="groundingdino_swint_ogc.pth",
                                                                   model_url="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"),
        Text2BoxVisualGroundingDinoModelTypes.SWIN_B: ModelDetails(repo_id="ShilongLiu/GroundingDINO",
                                                                   filename="groundingdino_swinb_cogcoor.pth",
                                                                   model_url="https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth")}

    def __init__(self,
                 weight_path: str = None,
                 width=800,
                 height=800,
                 mean_normalize=(0.485, 0.456, 0.406),
                 std_normalize=(0.229, 0.224, 0.225),
                 max_size=1333,
                 box_threshold=0.35,
                 text_threshold=0.25,
                 device: str = "cuda",
                 overwrite_weight_download=False,
                 model_type=Text2BoxVisualGroundingDinoModelTypes.SWIN_T):
        """
        Creates an instance of Text2ObjectVisualGroundingDinoTorch.
        :param weight_path: path to the weight file, if set to None, it will download the weight file from huggingface
        hub.
        :param width: width of the image
        :param height: height of the image. The height is not used and only width is used!
        :param mean_normalize: mean values to normalize the image.
        :param std_normalize: std values to normalize the image
        :param max_size: max size of the image.
        :param device: device to run the model on. Default is cuda.
        :param model_type: type of the model. It can be swin_b or swin_t. Default is swin_t which is smaller and faster.
        :param box_threshold: threshold for the boxes. Default is 0.35
        :param text_threshold: threshold for the text. Default is 0.25
        :param overwrite_weight_download: if set to True, it will download the weight file even if it exists.
        """
        self._weight_path = weight_path
        self._device = device
        self._width = width
        self._height = height
        self._max_size = max_size
        self._mean_normalize = mean_normalize
        self._std_normalize = std_normalize
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._load_model(device, model_type, weight_path, overwrite=overwrite_weight_download)
        self._transform = T.Compose(
            [
                T.RandomResize([self._width, self._height], max_size=self._max_size),
                T.ToTensor(),
                T.Normalize(list(self._mean_normalize), list(self._std_normalize)),
            ]
        )
        print("[INFO] Groundingdino Model loaded successfully!")

    def _load_model(self, device, model_type, weight_path, overwrite):
        if weight_path is not None:
            self._load_model_weight_path(device, model_type, weight_path)
        # else:
        #     self._model = self._load_model_hf(model_type, device=self._device)
        else:
            weight_path = DownloadUtils.download_file(Text2BoxVisualGroundingDino.MODEL_DETAILS[model_type].model_url,
                                                      exists_skip=not overwrite)
            self._load_model_weight_path(device, model_type, weight_path)

    def _load_model_weight_path(self, device, model_type, weight_path):
        if model_type == Text2BoxVisualGroundingDinoModelTypes.SWIN_B:
            args = SLConfig.fromfile(GroundingDINO_SwinB_cfg.__file__)
        elif model_type == Text2BoxVisualGroundingDinoModelTypes.SWIN_T:
            args = SLConfig.fromfile(GroundingDINO_SwinT_OGC.__file__)
        else:
            raise ValueError(f"model_type {model_type} is not supported, supported model_type are swin_b and swin_t")
        args.device = device
        self._model = build_model(args)
        checkpoint = torch.load(weight_path, map_location=self._device)
        self._model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self._model.eval().to(self._device)

    @staticmethod
    def _load_model_hf(model_type, device='cpu'):
        from huggingface_hub import hf_hub_download
        cache_config_file = hf_hub_download(repo_id=Text2BoxVisualGroundingDino.MODEL_DETAILS[model_type].repo_id,
                                            filename=Text2BoxVisualGroundingDino.MODEL_DETAILS[model_type].filename
                                            )

        args = SLConfig.fromfile(cache_config_file)
        model = build_model(args)
        args.device = device

        cache_file = hf_hub_download(repo_id=Text2BoxVisualGroundingDino.MODEL_DETAILS[model_type].repo_id,
                                     filename=Text2BoxVisualGroundingDino.MODEL_DETAILS[model_type].filename)
        checkpoint = torch.load(cache_file, map_location='cpu')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model

    @staticmethod
    def _preprocess_caption(caption: str) -> str:
        """
        Preprocess the caption to be used in the model.
        It makes the caption lower case and adds a dot at the end if it does not have one.
        :param caption:
        :return:
        """
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."

    def _predict(
            self,
            image: torch.Tensor,
            caption: str,
            box_threshold: float = 0.35,
            text_threshold: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Given an image and a caption, it returns the boxes, scores and classes.
        :param image:
        :param caption:
        :param box_threshold: threshold for the boxes. Default is 0.35
        :param text_threshold: threshold for the text. Default is 0.25
        :return:
        """
        caption = self._preprocess_caption(caption=caption)
        image = image.to(self._device)

        with torch.no_grad():
            outputs = self._model(image[None], captions=[caption])

        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

        mask = prediction_logits.max(dim=1)[0] > box_threshold
        logits = prediction_logits[mask]  # logits.shape = (n, 256)
        boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

        tokenizer = self._model.tokenizer
        tokenized = tokenizer(caption)

        phrases = [get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
                   for logit in logits]

        return boxes, logits.max(dim=1)[0], phrases

    def _process_pil(self, text: str,
                     frame: Image.Image,
                     threshold: Union[float, Tuple[float, float], None] = None) -> Text2BoxVisualGroundingDinoOutput:
        """
        Given a text and an image, it returns the boxes, scores and classes.
        :param text:
        :param frame:
        :return:
        """
        assert isinstance(frame, Image.Image), "frame must be a PIL image"
        if isinstance(threshold, float):
            text_threshold = threshold
            box_threshold = threshold
        elif isinstance(threshold, tuple) or isinstance(threshold, list):
            text_threshold, box_threshold = threshold
        else:
            text_threshold = self._text_threshold
            box_threshold = self._box_threshold

        h, w, _ = np.asarray(frame).shape
        frame, _ = self._transform(frame, None)
        boxes, scores, classes = self._predict(image=frame,
                                               caption=text,
                                               box_threshold=box_threshold,
                                               text_threshold=text_threshold)
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy().tolist()
        xyxy = Box.box2box(xyxy, in_source=Box.BoxSource.Torch, to_source=Box.BoxSource.Numpy,
                           shape_source=Box.PointSource.Numpy)
        output = Text2BoxVisualGroundingDinoOutput(boxes=xyxy, scores=scores.numpy().tolist(), labels=classes)
        return output

    def text_to_box(self, text: str, img: np.ndarray,
                    threshold: Union[float, Tuple[float, float], None] = None) -> Text2BoxVisualGroundingDinoOutput:
        """
        Given a text and an image, it returns the boxes, scores and classes.
        :param text:
        :param img:
        :param threshold:
        :return:
        """
        assert isinstance(img, np.ndarray) and len(img.shape) == 3, "frame must be a numpy array with 3 dimensions"
        frame = Image.fromarray(img)
        return self._process_pil(text=text, frame=frame, threshold=threshold)

    @staticmethod
    def annotate(img: np.ndarray, output: Text2BoxVisualGroundingDinoOutput, append_score=True,
                 copy_img=True) -> np.ndarray:
        """
        Given an image and the output of the model, it returns the image with the boxes and the text on it.
        :param output:
        :param img: input image
        :param append_score: if True, it appends the score to the label. Default is True.
        :param copy_img: if True, it returns a copy of the image. Default is True.
        :return:
        """
        if copy_img:
            img = img.copy()
        annotated_img = Box.put_box_text(img, output.boxes,
                                         label=[f"{l}" + f": {s:0.2f}" if append_score else "" for l, s in
                                                zip(output.labels, output.scores)])
        return annotated_img
