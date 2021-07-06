from deep_utils.utils.lib_utils.import_utils import import_module
from .main_resize import get_img_shape

resize = import_module("deep_utils.utils.resize_utils.main_resize", "resize")
cv2_resize = import_module("deep_utils.utils.resize_utils.main_resize", "cv2_resize")
