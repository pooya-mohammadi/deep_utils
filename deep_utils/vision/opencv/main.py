def show_destroy_cv2(img, win_name=''):
    import cv2
    try:
        cv2.imshow(win_name, img)
        cv2.waitKey(0)
        cv2.destroyWindow(win_name)
    except Exception as e:
        cv2.destroyWindow(win_name)
        raise e
