# Yolov7 Notes
2) copied box_iou from general to metrics, so that I could prevent the circular import
2) move from .models.experimental import attempt_load below FILE include
3) manually changed the following in google utils: 
```commandline
# tag = subprocess.check_output('git tag', shell=True).decode().split()[-1]
tag = "v0.1"
```
4) I had to manually remove yolov5 from the sys.path in the load_model method