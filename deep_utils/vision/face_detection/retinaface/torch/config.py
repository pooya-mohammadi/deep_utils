class Config:
    device = 'cpu'
    min_face_size = 20.0
    thresholds = (0.6, 0.7, 0.8)
    min_detection_size = 12
    factor = 0.707
    top_k = 5000
    nms_thresholds = 0.4
    keep_top_k = 750
    visualization_threshold = 0.6
    resnet50_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.2.0/resnet50.pth"
    resnet50_cache = 'weights/vision/face_detection/retinaface/torch/resnet50.pth'
    resnet50 = None
    mobilenet_url = "https://github.com/Practical-AI/deep_utils/releases/download/0.2.0/mobilenet.pth"
    mobilenet_cache = 'weights/vision/face_detection/retinaface/torch/mobilenet.pth'
    mobilenet = None
    mobilenetV1X_url = 'https://github.com/Practical-AI/deep_utils/releases/download/0.2.0/mobilenetV1X0.25_pretrain.tar'
    mobilenetV1X_cache = 'weights/vision/face_detection/retinaface/torch/mobilenetV1X0.25_pretrain.tar'
    mobilenetV1X = None
    network = 'mobilenet'
    cfg_mnet = {
        'name': 'mobilenet',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 32,
        'ngpu': 1,
        'epoch': 250,
        'decay1': 190,
        'decay2': 220,
        'image_size': 640,
        'pretrain': True,
        'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
        'in_channel': 32,
        'out_channel': 64
    }

    cfg_re50 = {
        'name': 'resnet50',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
        'loc_weight': 2.0,
        'gpu_train': True,
        'batch_size': 24,
        'ngpu': 4,
        'epoch': 100,
        'decay1': 70,
        'decay2': 90,
        'image_size': 840,
        'pretrain': True,
        'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
        'in_channel': 256,
        'out_channel': 256
    }
