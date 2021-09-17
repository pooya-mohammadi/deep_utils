import numpy as np
from deep_utils.vision.face_detection.main.main_face_detection import FaceDetector
from deep_utils.utils.lib_utils.lib_decorators import get_from_config, expand_input, get_elapsed_time, rgb2bgr
from deep_utils.utils.lib_utils.download_utils import download_decorator
from deep_utils.utils.box_utils.boxes import Box, Point
from .config import Config
import sys


class RetinaFaceTorchFaceDetector(FaceDetector):
    def __init__(self, **kwargs):
        import torch
        from .src.layers.functions.prior_box import PriorBox
        from .src.utils.nms.py_cpu_nms import py_cpu_nms
        from .src.retinaface import RetinaFace
        from .src.utils.box_utils import decode, decode_landm, _preprocess
        from .src.load_model import load_model
        self.RetinaFace = RetinaFace
        self.load_model_weights = load_model
        self.torch = torch
        self.PriorBox = PriorBox
        self.py_cpu_nms = py_cpu_nms
        self.decode = decode
        self.decode_landm = decode_landm
        self._preprocess = _preprocess
        self.resize = 1
        self.pretrained_mobilenet = None
        super().__init__(name=self.__class__.__name__,
                         file_path=__file__,
                         download_variables=("resnet50", "mobilenet", "mobilenetV1X"),
                         **kwargs)
        self.config: Config

    @download_decorator
    def load_model(self):
        # LOAD MODELS
        cfg_resnet = Config.cfg_re50
        cfg_mobilenet = Config.cfg_mnet
        resnet50_path = self.config.resnet50
        mobilenet_path = self.config.mobilenet
        self.pretrained_mobilenet = self.config.mobilenetV1X
        return dict(resnet50=(cfg_resnet, resnet50_path), mobilenet=(cfg_mobilenet, mobilenet_path))

    @get_from_config
    @get_elapsed_time
    @rgb2bgr('bgr')
    @expand_input(3)
    def detect_faces(self,
                     images,
                     is_rgb,
                     network,
                     resize_size=None,
                     img_scale_factor=None,
                     img_mean=None,
                     resize_mode=None,
                     confidence=None,
                     get_time=False,
                     round_prec=4):
        network = str.lower(network)
        load_to_cpu = True
        device = self.config.device
        if device == 'cuda':
            load_to_cpu = False
        if network == 'resnet50':
            cfg, model_path = self.load_model()['resnet50']
            net = self.RetinaFace(cfg=cfg, phase='test')
            net = self.load_model_weights(net, model_path, load_to_cpu=load_to_cpu)
        elif network == 'mobilenet':
            cfg, model_path = self.load_model()['mobilenet']
            net = self.RetinaFace(cfg=cfg, phase='test', pretrained_model=self.pretrained_mobilenet)
            net = self.load_model_weights(net, model_path, load_to_cpu=load_to_cpu)
        else:
            print("The network backbone is wrong!")
            sys.exit(1)

        if images.ndim < 4:
            images = np.array([images])
        net.eval()
        net.to(device)
        face_points = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
        # for img_n in range(images.shape[0]):
        #     img = np.float32(images[img_n])
        # img = self.torch.FloatTensor(np.concatenate(images)).to(self.config.device)

        img = self.torch.FloatTensor(self._preprocess(images)).to(device)
        _, im_height, im_width = img[0].shape
        scale = self.torch.Tensor([img[0].shape[2], img[0].shape[1], img[0].shape[2], img[0].shape[1]])
        # im_height, im_width, _ = img.shape
        # scale = self.torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        # img -= (104, 117, 123)
        # img = img.transpose(2, 0, 1)
        # img = self.torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        loc, conf, landms = net(img)

        priorbox = self.PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        loc_data = loc.data.squeeze(0)
        boxes = self.decode(loc_data, prior_data, cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = self.decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = self.torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                    img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                    img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.config.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        bboxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.py_cpu_nms(bboxes, self.config.nms_thresholds)
        bboxes = bboxes[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        bboxes = bboxes[:self.config.keep_top_k, :]
        landms = landms[:self.config.keep_top_k, :]

        boxes_, confidences_, landmarks_ = [], [], []
        for num in range(len(bboxes)):
            box_tensor = bboxes[num, :]
            boxes = [box_tensor[0], box_tensor[1], box_tensor[2], box_tensor[3]]
            boxes = Box.box2box(boxes, in_source=Box.BoxSource.Torch, to_source=Box.BoxSource.Numpy)
            boxes_.append(boxes)
            confidences_.append(scores.round(round_prec))
            if len(landms) != 0:
                landmarks = [[Point.point2point((landms[j][2 * i], landms[j][2 * i + 1]),
                                                in_source='Torch', to_source='Numpy') for i in range(5)] for j in
                             range(len(landms))]
            img_landmarks = []
            for i in range(len(landmarks)):
                face_dict = {}
                for points, face in zip(landmarks[i], face_points):
                    face_dict[face] = points
                img_landmarks.append(face_dict)
            landmarks_.append(img_landmarks)

        output = self.output_class(boxes=boxes_, confidences=confidences_, landmarks=landmarks_[00])
        return output
