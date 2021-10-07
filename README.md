[![Downloads](https://static.pepy.tech/badge/deep_utils)](https://pepy.tech/project/deep_utils) [![PyPI](https://img.shields.io/pypi/v/deep_utils.svg)](https://pypi.python.org/pypi/deep_utils)
# Deep Utils 

This repository contains the most frequently used deep learning modules and functions.


## Table of contents

* [Table of contents](#table-of-contents)
* [Quick start](#quick-start)
* [References](#references)

## Quick start

1. Install:
    
    ```bash
    # With pip:
    pip install deep_utils
    
    # or from the repo
    pip install git+https://github.com/Practical-AI/deep_utils.git
   
    # or clone the repo
    git clone https://github.com/Practical-AI/deep_utils.git deep_utils
    pip install -U deep_utils 
   ```
    
1. In python, import deep_utils and instantiate models:
    
    ```python
    from deep_utils import face_detector_loader, list_face_detection_models
    
   # list all the available models first 
   list_face_detection_models()
   
   # Create a face detection model using SSD
   face_detector = face_detector_loader('SSDCV2CaffeFaceDetector')
    
    
1. Detect an image:

    ```python
    import cv2
    from deep_utils import show_destroy_cv2, Box
    
    # Load an image
    img = cv2.imread(<image path>)

    # Detect the faces
    boxes, confidences = face_detector.detect_faces(img)
    
    # Draw detected boxes on the image 
    img = Box.put_box(img, boxes)
    
    # show the results
    show_destroy_cv2(img) 
    ```
## References

1. Tim Esler's facenet-pytorch repo: [https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)
