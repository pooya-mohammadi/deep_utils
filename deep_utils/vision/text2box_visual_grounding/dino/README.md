# Text to Box Visual Grounding

# Installation
```commandline
pip install git+https://github.com/IDEA-Research/GroundingDINO
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

model = Text2BoxVisualGroundingDino()

img_path = "golsa_in_garden.jpg"
img = np.asarray(Image.open(img_path))

output = model.text_to_box(text="Hen", img=img)
print(output.boxes, output.scores, output.labels)
annotated_img = model.annotate(img, output)
plt.axis("off")
plt.imshow(annotated_img)
```

Input Image:</br>
<img src="https://github.com/pooya-mohammadi/deep_utils/releases/download/1.0.2/golsa_in_garden.jpg" width="400">

Output Image:</br>
<img src="https://github.com/pooya-mohammadi/deep_utils/releases/download/1.0.2/golsa_in_garden_dino.png" width="400">