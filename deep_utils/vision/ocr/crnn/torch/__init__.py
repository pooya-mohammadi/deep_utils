try:
    from deep_utils.dummy_objects.vision.ocr.crnn.torch import CRNNModelTorch
    from .crnn_model import CRNNModelTorch
except:
    pass

try:
    from deep_utils.dummy_objects.vision.ocr.crnn.torch import CRNNInferenceTorch
    from .crnn_inference import CRNNInferenceTorch
except:
    pass
