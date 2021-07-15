import cv2
from deep_utils import Box, show_destroy_cv2, object_detector_loader

if __name__ == '__main__':
    model = object_detector_loader("YOLOV5TorchObjectDetector",
                                   model_weight='/home/ai/.cache/torch/hub/ultralytics_yolov5_master/yolov5s.pt')
    img = cv2.imread('../data/movie-stars.jpg')
    res = model.detect_objects(img, is_rgb=False, get_time=True)
    img = Box.put_box(img, res['boxes'])
    show_destroy_cv2(img)
    print(res)
