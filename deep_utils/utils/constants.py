import os
from enum import Enum

DUMMY_PATH = "dummy_objects.dummies"
LIB_NAME = "deep_utils"


class Backends(str, Enum):
    """
    All required dependency packages and libraries. Note that the values here must be the exact module names used
    for importing, for example if you set PILLOW the value must be `PIL` not `pillow`, `pil`, etc.
    """
    MINIO = "minio"
    NUMPY = "numpy"
    TORCH = "torch"
    TRANSFORMERS = "transformers"
    DATASETS = "datasets"
    TOKENIZERS = "tokenizers"
    SOUNDFILE = "soundfile"
    LIBROSA = "librosa"
    WANDB = "wandb"
    GENSIM = "gensim"
    PILLOW = "PIL"
    JIWER = "jiwer"
    NLTK = "nltk"
    SCIKIT = "sklearn"
    SEQEVAL = "seqeval"
    SIMPLE_ITK = "SimpleITK"
    TENSORFLOW = "tensorflow"
    QDRANT_CLIENT = "qdrant_client"
    REQUESTS = "requests"
    CV2 = "cv2"
    GLIDE_TEXT2IM = "glide_text2im"
    GROUNDINGDINO = "groundingdino"
    MONAI = "monai"
    TORCHVISION = "torchvision"
    TIMM = "timm"
    FAIRSCALE = "fairscale"
    ELASTICSEARCH = "elasticsearch"
    ALBUMENTATIONS = 'albumentations'
    AIOHTTP = "aiohttp"
    NIBABEL = "nibabel"
    TORCHAUDIO = "torchaudio"
    TikToken = "tiktoken"
    DECORD = "decord"

    def __str__(self):
        return str(self.value)
