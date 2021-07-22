import cv2
import numpy as np
from deep_utils import Box, show_destroy_cv2, resize
from deep_utils import face_detector_loader
if __name__ == '__main__':
    face_detector = face_detector_loader('SSDCV2CaffeFaceDetector')
    img_2 = cv2.imread('../data/movie-stars.jpg')
    img_1 = cv2.imread('/home/ai/projects/mtcnn-pytorch/images/office1.jpg')
    img_2 = resize(img_2, img_1.shape[:2])
    images = np.array([img_2, img_1])
    res = face_detector.detect_faces(images, is_rgb=False, get_time=True)
    for i, img in enumerate(images):
        img = Box.put_box(img, res['boxes'][i])
        print("boxes:", res['boxes'][i], "confidences: ", res['confidences'][i], "elapsed_time: ", res['elapsed_time'])
        show_destroy_cv2(img)
