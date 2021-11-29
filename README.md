[![Downloads](https://static.pepy.tech/badge/deep_utils)](https://pepy.tech/project/deep_utils) [![PyPI](https://img.shields.io/pypi/v/deep_utils.svg)](https://pypi.python.org/pypi/deep_utils)
<div id="top"></div>
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/pooya-mohammadi/deep_utils">
    <img src="https://raw.githubusercontent.com/pooya-mohammadi/deep_utils/master/images/logo/deep_utils.png" alt="Logo">
  </a>

<h3 align="center">Deep Utils</h3>

  <p align="center">
    A toolkit for deep-learning practitioners!

</div>

This repository contains the most frequently used deep learning models and functions. **Deep_Utils** is still under
heavy development, so please remember that many things may change. Please install the latest version using pypi.

## Table of contents

* [About The Project](#about-the-project)
* [Installation](#installation)
* [Vision](#vision)
    * [Face Detection](#face-detection)
        * [Quick Start](#quick-start)
        * [MTCNN](#mtcnn)
    * [Object Detection](#object-detection)
* [References](#references)

<p align="right">(<a href="#top">back to top</a>)</p>

## About the Project
Many deep learning toolkits are available on GitHub; however, we didn't find one that suited our needs.
So, we created this enhanced one. This toolkit minimizes the deep learning teams' coding effort to utilize the functionalities of famous deep learning models such as MTCNN in face detection, yolov5 in object detection, and many other repositories and models in various fields. In addition, it provides functions for preprocessing, monitoring, and manipulating datasets that can come in handy in any programming project.

**What we have done so far:**
* The outputs of all the models are standard numpy
* Single predict and batch predict of all models are ready
* handy functions and tools are tested and ready to use

<p align="right">(<a href="#top">back to top</a>)</p>

## Installation:
```bash
    # pip: recommended
    pip install -U deep_utils
    
    # repository
    pip install git+https://github.com/pooya-mohammadi/deep_utils.git
   
    # clone the repo
    git clone https://github.com/pooya-mohammadi/deep_utils.git deep_utils
    pip install -U deep_utils 
```
<p align="right">(<a href="#top">back to top</a>)</p>

## Face Detection

We have gathered a rich collection of face detection models that are as follows. If you need a model that we don't have, please feel free to open an issue or create a pull request.

### MTCNN-Torch

1. After Installing the library, import deep_utils and instantiate models:

```python
from deep_utils import face_detector_loader, list_face_detection_models
    
# This line will print all the available models 
list_face_detection_models()
   
# Create a face detection model using MTCNN-Torch
face_detector = face_detector_loader('MTCNNTorchFaceDetector')
```
2. The model is instantiated, Now let's Detect an image:

```python
import cv2
from deep_utils import show_destroy_cv2, Box, download_file

# Download an image
download_file("https://raw.githubusercontent.com/pooya-mohammadi/deep_utils/master/examples/vision/data/movie-stars.jpg")

# Load an image
img = cv2.imread("movie-starts.jpg")

# show the image. Press a button to proceed
show_destroy_cv2(img)

# Detect the faces
boxes, confidences = face_detector.detect_faces(img)
    
# Draw detected boxes on the image. 
img = Box.put_box(img, boxes)
    
# show the results
show_destroy_cv2(img) 
```
The result:

<img src="https://raw.githubusercontent.com/pooya-mohammadi/deep_utils/master/examples/vision/data/movie-starts-mtccn-torch.jpg" alt="Logo" >
<p align="right">(<a href="#top">back to top</a>)</p>
<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this toolkit enhanced, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

## 🤝 Collaborators

<table>
  <tr>
    <td align="center">
      <a href="#">
        <img src="https://avatars.githubusercontent.com/u/55460936?v=4" width="100px;" alt="Pooya Mohammadi no GitHub"/><br>
        <sub>
          <b>Pooya Mohammadi Kazaj</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Vargha-Kh">
        <img src="https://avatars.githubusercontent.com/u/69680607?v=4" width="100px;" alt="Vargha Khallokhi"/><br>
        <sub>
          <b>Vargha Khallokhi</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/dornasabet">
        <img src="https://avatars.githubusercontent.com/u/74057278?v=4" width="100px;" alt="Dorna Sabet"/><br>
        <sub>
          <b>Dorna Sabet</b>
        </sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/alirezakazemipour">
        <img src="https://avatars.githubusercontent.com/u/32295763?v=4" width="100px;" alt="Alireza Kazemipour"/><br>
        <sub>
          <b>Alireza Kazemipour</b>
        </sub>
      </a>
    </td>
  </tr>
</table>
<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Pooya Mohammadi:
* LinkedIn [www.linkedin.com/in/pooya-mohammadi](www.linkedin.com/in/pooya-mohammadi)
* Email: [pooyamohammadikazaj@gmail.com](pooyamohammadikazaj@gmail.com)

Project Link: [https://github.com/pooya-mohammadi/deep_utils](https://github.com/your_username/repo_name)

<p align="right">(<a href="#top">back to top</a>)</p>



## References

1. Tim Esler's facenet-pytorch
   repo: [https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)
