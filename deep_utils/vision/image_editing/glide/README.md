# GLIDE Image Editing


# Installation

```bash
pip install git+https://github.com/openai/glide-text2im
```

#### Download a sample image

```commandline
wget -q https://github.com/pooya-mohammadi/deep_utils/releases/download/1.0.2/golsa_in_garden.jpg
```

**Input Image**:<br/>
<img src="https://github.com/pooya-mohammadi/deep_utils/releases/download/1.0.2/golsa_in_garden.jpg" width="400">

```python
import matplotlib.pyplot as plt
from deep_utils import ImageEditingGLIDE
from PIL import Image

pil_img = Image.open("golsa_in_garden.jpg")
# position of the editing box. Here the hen in the image. The  
box = [340.6672668457031, 403.7683410644531, 372.0812072753906, 439.3288879394531]
glide_model = ImageEditingGLIDE()
text = "dead leaves"
edited_image = glide_model.edit_box(pil_img, text=text, box=box)
plt.imshow(edited_image)
```

**Output Image**: The `hen` is removed and replaced with `dead leaves` as the background<br/>
<img src="https://github.com/pooya-mohammadi/deep_utils/releases/download/1.0.2/glide_output.jpg" width="400">

**Note:** The best way to get the box is to use the `Text2BoxVisualGroundingDino` model. See the example in the previous
section. Or check the following full sample [easy_image_editing](https://github.com/pooya-mohammadi/easy_image_editing)