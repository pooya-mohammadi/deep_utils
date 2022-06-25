from itertools import product as product
from math import ceil

import numpy as np
import torch


class PriorBox:
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg["min_sizes"]
        self.steps = cfg["steps"]
        self.clip = cfg["clip"]
        self.name = "s"

    def forward(self, image_size):
        anchors = []
        feature_maps = [
            [ceil(image_size[0] / step), ceil(image_size[1] / step)]
            for step in self.steps
        ]
        for k, f in enumerate(feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / image_size[1]
                    s_ky = min_size / image_size[0]
                    dense_cx = [x * self.steps[k] / image_size[1]
                                for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / image_size[0]
                                for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
