[![Downloads](https://static.pepy.tech/badge/deep_utils)](https://pepy.tech/project/deep_utils) [![PyPI](https://img.shields.io/pypi/v/deep_utils.svg)](https://pypi.python.org/pypi/deep_utils) [![build](https://github.com/pooya-mohammadi/deep_utils/actions/workflows/automatic-release.yml/badge.svg)](https://github.com/pooya-mohammadi/deep_utils/actions/workflows/automatic-release.yml)

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
heavy development, so take into consideration that many features may change in the future. Install the latest version using pypi.

## Table of contents

* [About The Project](#about-the-project)
* [Installation](#installation)
* [Vision](#vision)
    * [Face Detection](#face-detection)
        * [MTCNN](#mtcnn)
* [Utils](#utils)
  * [DictNamedTuple](#dictnametuple)
* [Contributing](#Contributing)
* [Licence](#Licence)
* [Collaborators](#Collaborators)
* [Contact](#Contact)
* [References](#references)
* [Citation](#citation)

<p align="right">(<a href="#top">back to top</a>)</p>

## About the Project
Many deep learning toolkits are available on GitHub; however, we couldn't find one that would suit our needs.
So, we created this improved one. This toolkit minimizes the deep learning teams' coding efforts to utilize the functionalities of famous deep learning models such as MTCNN in face detection, yolov5 in object detection, and many other repositories and models in various fields. In addition, it provides functionalities for preprocessing, monitoring, and manipulating datasets that can come in handy in any programming project.

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

# Vision
We support two subsets of models in Computer Vision. 
* Face Detection
* Object Detection

## Face Detection

We have gathered a rich collection of face detection models which are mentioned in the following list. If you notice any model missing, feel free to open an issue or create a pull request.

### MTCNN
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pooya-mohammadi/Face/blob/master/_02_mtcnn_tf1/deep_utils_mtcnn.ipynb)
1. After Installing the library, import deep_utils and instantiate models:

```python
from deep_utils import face_detector_loader, list_face_detection_models
    
# This line will print all the available models 
print(list_face_detection_models())
   
# Create a face detection model using MTCNN-Torch
face_detector = face_detector_loader('MTCNNTorchFaceDetector')
```
2. The model is instantiated, Now let's Detect an image:

```python
import cv2
from deep_utils import show_destroy_cv2, Box, download_file, Point

# Download an image
download_file("https://raw.githubusercontent.com/pooya-mohammadi/deep_utils/master/examples/vision/data/movie-stars.jpg")

# Load an image
img = cv2.imread("movie-stars.jpg")

# show the image. Press a button to proceed
show_destroy_cv2(img)

# Detect the faces
result = face_detector.detect_faces(img, is_rgb=False)

# Draw detected boxes on the image.
img = Box.put_box(img, result.boxes)

# Draw the landmarks
for landmarks in result.landmarks:
    Point.put_point(img, list(landmarks.values()), radius=3)

# show the results
show_destroy_cv2(img)
```
The result:

<img src="https://raw.githubusercontent.com/pooya-mohammadi/deep_utils/master/examples/vision/data/movie-starts-mtccn-torch.jpg" alt="Logo" >
<p align="right">(<a href="#top">back to top</a>)</p>

## Utils


### DictNamedTuple
In this custom data type, we have added the methods of the Dict type to the NamedTuple type. You have access to .get(), .values(), .items() alongside all of the functionalities of a NamedTuple. Also, all the outputs of our models are DictNamedTuple, and you can modify and manipulate them easily. Let's see how to use it:

```
from deep_utils import dictnamedtuple
# create a new object
dict_object = dictnamedtuple(typename='letters', field_names=['firstname', 'lastname'])
# pass the values
instance_dict = dict_object(firstname='pooya', lastname='mohammadi')

# get items and ...
print("items: ", instance_dict.items())
print("keys: ", instance_dict.keys())
print("values: ", instance_dict.values())
print("firstname: ", instance_dict.firstname)
print("firstname: ", instance_dict['firstname'])
print("lastname: ", instance_dict.lastname)
print("lastname: ", instance_dict['lastname'])
```

```
# results
items:  [('firstname', 'pooya'), ('lastname', 'mohammadi')]
keys:  ['firstname', 'lastname']
values:  ['pooya', 'mohammadi']
firstname:  pooya
firstname:  pooya
lastname:  mohammadi
lastname:  mohammadi
```

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this toolkit enhanced, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a ⭐️! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.
The LICENSE of each model is located inside its corresponding directory. 

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
      <a href="https://github.com/MenuaB">
        <img src="https://avatars.githubusercontent.com/u/16352202?v=4" width="100px;" alt="Menua Bedrosian"/><br>
        <sub>
          <b>Menua Bedrosian </b>
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

<p align="right">(<a href="#top">back to top</a>)</p>

## Citation

Please cite deep-utils if it helps your research. You can use the following BibTeX entry:
```
@misc{deep_utils,
	title = {deep_utils},
	author = {Mohammadi Kazaj, Pooya},
	howpublished = {\url{github.com/pooya-mohammadi/deep_utils}},
	year = {2021}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>
