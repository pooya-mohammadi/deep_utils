import cv2
from deep_utils.vision.object_detection.yolo.v5.torch.yolo_v5_torch_object_detection import YOLOV5TorchObjectDetection
from deep_utils import Box, show_destroy_cv2

if __name__ == '__main__':
    model = YOLOV5TorchObjectDetection(model_weight='/home/ai/.cache/torch/hub/ultralytics_yolov5_master/yolov5s.pt')
    img = cv2.imread('../data/movie-stars.jpg')
    res = model.detect_objects(img, is_rgb=False)
    img = Box.put_boxes(img, res['boxes'])
    show_destroy_cv2(img)
    print(res)
