# Text to Box Visual Grounding

# Installation
```commandline
pip install git+https://github.com/IDEA-Research/GroundingDINO
```

# Download weights
#### First Download the weights:
```commandline
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```
#### Download a sample image
```commandline
wget -q https://github.com/pooya-mohammadi/deep_utils/releases/download/1.0.2/golsa_in_garden.jpg
```

## Usage
The `Text2BoxVisualGroundingDino` class can be instantiated as testes as follows:
```python
from PIL import Image
from deep_utils import Text2BoxVisualGroundingDino
import numpy as np
import matplotlib.pyplot as plt

weight_path = "groundingdino_swint_ogc.pth"
model = Text2BoxVisualGroundingDino(weight_path=weight_path)

img_path = "golsa_in_garden.jpg"
img = np.asarray(Image.open(img_path))

output = model.text_to_box(text="Hen", img=img)
print(output.boxes, output.scores, output.labels)
annotated_img = model.annotate(img, output)
plt.axis("off")
plt.imshow(annotated_img)
```

