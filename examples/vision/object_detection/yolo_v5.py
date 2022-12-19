import cv2

from deep_utils import Box, YOLOV5TorchObjectDetector, show_destroy_cv2

if __name__ == "__main__":
    model = YOLOV5TorchObjectDetector()
    img = cv2.imread("../data/movie-starts-mtccn-torch.jpg")
    res = model.detect_objects(img, is_rgb=False, get_time=True)
    img = Box.put_box_text(img, res.boxes, res.class_names)
    show_destroy_cv2(img)
    print(res)
