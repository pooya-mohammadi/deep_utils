import cv2

from deep_utils import Box, YOLOV5TorchObjectDetector, show_destroy_cv2

if __name__ == "__main__":
    model = YOLOV5TorchObjectDetector(
        model_weight="/home/ai/projects/Yolov5_DeepSort_Pytorch/yolov5s.pt"
    )
    img = cv2.imread("../data/movie-stars.jpg")
    res = model.detect_objects(img, is_rgb=False, get_time=True)
    img = Box.put_box_text(img, res.boxes, res.class_names)
    show_destroy_cv2(img)
    print(res)
