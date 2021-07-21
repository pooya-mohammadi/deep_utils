import cv2
import numpy as np
from deep_utils import Box, show_destroy_cv2
from deep_utils import face_detector_loader

if __name__ == '__main__':
    face_detector = face_detector_loader('MTCNNTorchFaceDetector')
    img = cv2.imread('../data/movie-stars.jpg')
    res = face_detector.detect_faces(np.array([cv2.imread('../data/movie-stars.jpg'), cv2.imread('/home/ai/projects/mtcnn-pytorch/images/office1.jpg')]), is_rgb=False, get_time=True)
    img = Box.put_box(img, res['boxes'])
    print("boxes:", res['boxes'], "confidences: ", res['confidences'], "elapsed_time: ", res['elapsed_time'])
    show_destroy_cv2(img)
