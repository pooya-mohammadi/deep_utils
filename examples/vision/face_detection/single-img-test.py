import cv2
from deep_utils import Box, show_destroy_cv2, Point
from deep_utils import face_detector_loader

if __name__ == '__main__':
    face_detector = face_detector_loader('MTCNNTorchFaceDetector')
    img = cv2.imread('../data/movie-stars.jpg')
    res = face_detector.detect_faces(img, is_rgb=False, get_time=True, confidence=0.99)
    for points in res['landmarks']:
        Point.put_point(img, points, radius=3)
    img = Box.put_box(img, res['boxes'])
    print("boxes:", res['boxes'], "confidences: ", res['confidences'], "elapsed_time: ", res['elapsed_time'])
    show_destroy_cv2(img)
