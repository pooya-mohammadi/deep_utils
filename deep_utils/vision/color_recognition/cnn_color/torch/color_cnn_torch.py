from copy import deepcopy

import torch
from torch import nn

from deep_utils.blocks.torch.blocks_torch import BlocksTorch


class ColorRecognitionCNNTorch(nn.Module):
    def __init__(self, n_classes, in_channel=3):
        super(ColorRecognitionCNNTorch, self).__init__()
        # ============================================= TOP BRANCH ===================================================

        self.top_conv1 = BlocksTorch.conv_norm_act(in_channel, out_c=48, k=(11, 11), s=(4, 4), norm="bn", pooling="max",
                                                   pooling_k=(3, 3), pooling_s=(2, 2))
        self.top_top_conv2 = BlocksTorch.conv_norm_act(24, out_c=64, k=(3, 3), s=(1, 1), padding="same",
                                                       norm="bn", pooling="max", pooling_k=(3, 3), pooling_s=(2, 2))
        self.top_bot_conv2 = deepcopy(self.top_top_conv2)

        self.top_conv3 = BlocksTorch.conv_norm_act(128, out_c=192, k=(3, 3), s=(1, 1), padding="same")

        self.top_top_conv4 = BlocksTorch.conv_norm_act(96, out_c=96, k=(3, 3), s=(1, 1), padding="same")
        self.top_bot_conv4 = deepcopy(self.top_top_conv4)

        self.top_top_conv5 = BlocksTorch.conv_norm_act(96, out_c=64, k=(3, 3), s=(1, 1), padding="same", pooling="max",
                                                       pooling_k=(3, 3), pooling_s=(2, 2))
        self.top_bot_conv5 = deepcopy(self.top_top_conv5)

        # ============================================= TOP BOTTOM ===================================================
        self.bottom_conv1 = deepcopy(self.top_conv1)
        self.bottom_top_conv2 = deepcopy(self.top_top_conv2)
        self.bottom_bot_conv2 = deepcopy(self.top_bot_conv2)
        self.bottom_conv3 = deepcopy(self.top_conv3)
        self.bottom_top_conv4 = deepcopy(self.top_top_conv4)
        self.bottom_bot_conv4 = deepcopy(self.top_bot_conv4)
        self.bottom_top_conv5 = deepcopy(self.top_top_conv5)
        self.bottom_bot_conv5 = deepcopy(self.top_bot_conv5)

        # ============================ CONCATENATE TOP AND BOTTOM BRANCH & Fully connected layers ====================
        self.fc1 = BlocksTorch.fc_dropout(6400, out_f=4096, drop_p=0.6)
        self.fc2 = BlocksTorch.fc_dropout(4096, out_f=4096, drop_p=0.6)
        self.output = BlocksTorch.fc_dropout(4096, out_f=n_classes, act=None)

    def forward(self, input_images):
        # ============================================= TOP Branch ===================================================
        top_conv1 = self.top_conv1(input_images)

        top_top_conv1, top_bot_conv1 = torch.split(top_conv1, top_conv1.size(1) // 2, dim=1)
        top_top_conv2 = self.top_top_conv2(top_top_conv1)
        top_bot_conv2 = self.top_bot_conv2(top_bot_conv1)

        top_conv2 = torch.concat([top_top_conv2, top_bot_conv2], dim=1)
        top_conv3 = self.top_conv3(top_conv2)

        top_top_conv3, top_bot_conv3 = torch.split(top_conv3, top_conv3.size(1) // 2, dim=1)
        top_top_conv4 = self.top_top_conv4(top_top_conv3)
        top_bot_conv4 = self.top_bot_conv4(top_bot_conv3)

        top_top_conv5 = self.top_top_conv5(top_top_conv4)
        top_bot_conv5 = self.top_bot_conv5(top_bot_conv4)
        # ============================================= TOP BOTTOM ===================================================
        bottom_conv1 = self.bottom_conv1(input_images)

        bottom_top_conv1, bottom_bot_conv1 = torch.split(bottom_conv1, bottom_conv1.size(1) // 2, dim=1)
        bottom_top_conv2 = self.bottom_top_conv2(top_top_conv1)
        bottom_bot_conv2 = self.bottom_bot_conv2(top_bot_conv1)

        bottom_conv2 = torch.concat([bottom_top_conv2, bottom_bot_conv2], dim=1)
        bottom_conv3 = self.top_conv3(bottom_conv2)

        bottom_top_conv3, bottom_bot_conv3 = torch.split(bottom_conv3, bottom_conv3.size(1) // 2, dim=1)
        bottom_top_conv4 = self.bottom_top_conv4(bottom_top_conv3)
        bottom_bot_conv4 = self.bottom_bot_conv4(bottom_bot_conv3)

        bottom_top_conv5 = self.bottom_top_conv5(bottom_top_conv4)
        bottom_bot_conv5 = self.bottom_bot_conv5(bottom_bot_conv4)

        # ======================================== CONCATENATE TOP AND BOTTOM BRANCH =================================
        conv_output = torch.concat([top_top_conv5, top_bot_conv5, bottom_top_conv5, bottom_bot_conv5], dim=1)
        flatten = torch.flatten(conv_output, start_dim=1)
        fc1 = self.fc1(flatten)
        fc2 = self.fc2(fc1)
        logits = self.output(fc2)
        return logits
