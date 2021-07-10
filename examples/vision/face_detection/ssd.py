import cv2
from deep_utils import Box, show_destroy_cv2
from deep_utils import face_detector_loader

if __name__ == '__main__':
    face_detector = face_detector_loader('SSDCV2CaffeFaceDetector')
    img = cv2.imread('../data/movie-stars.jpg')
    boxes, confidences, elapsed_time = face_detector.detect_faces(img, get_time=True)
    img = Box.put_boxes(img, boxes)
    print("boxes:", boxes, "confidences: ", confidences, "elapsed_time: ", elapsed_time)
    show_destroy_cv2(img)
