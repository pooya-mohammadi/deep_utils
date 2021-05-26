import cv2


def show_destroy(img, win_name=''):
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyWindow(win_name)
