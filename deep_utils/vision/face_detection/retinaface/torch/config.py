class Config:
    def __init__(self):
        self.device = "cpu"
        self.min_face_size = 20.0
        self.thresholds = (0.6, 0.7, 0.8)
        self.min_detection_size = 12
        self.factor = 0.707
        self.top_k = 5000
        self.nms_thresholds = 0.4
        self.keep_top_k = 750
        self.visualization_threshold = 0.6
        self.resnet50_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.2.0/resnet50.pth"
        self.resnet50_cache = (
            "weights/vision/face_detection/retinaface/torch/resnet50.pth"
        )
        self.resnet50 = None
        self.mobilenet_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.2.0/mobilenet.pth"
        self.mobilenet_cache = (
            "weights/vision/face_detection/retinaface/torch/mobilenet.pth"
        )
        self.mobilenet = None
        self.mobilenetV1X_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.2.0/mobilenetV1X0.25_pretrain.tar"
        self.mobilenetV1X_cache = "weights/vision/face_detection/retinaface/torch/mobilenetV1X0.25_pretrain.tar"
        self.mobilenetV1X = None
        self.network = "mobilenet"
        self.cfg_mnet = {
            "name": "mobilenet",
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
            "variance": [0.1, 0.2],
            "clip": False,
            "loc_weight": 2.0,
            "gpu_train": True,
            "batch_size": 32,
            "ngpu": 1,
            "epoch": 250,
            "decay1": 190,
            "decay2": 220,
            "image_size": 640,
            "pretrain": True,
            "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
            "in_channel": 32,
            "out_channel": 64,
        }

        self.cfg_re50 = {
            "name": "resnet50",
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
            "variance": [0.1, 0.2],
            "clip": False,
            "loc_weight": 2.0,
            "gpu_train": True,
            "batch_size": 24,
            "ngpu": 4,
            "epoch": 100,
            "decay1": 70,
            "decay2": 90,
            "image_size": 840,
            "pretrain": True,
            "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
            "in_channel": 256,
            "out_channel": 256,
        }
