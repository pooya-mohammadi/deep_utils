import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_utils.blocks.torch.blocks_torch import BlocksTorch


class CRNNModelTorch(nn.Module):

    def __init__(self, img_h, n_channels, n_classes, n_hidden, lstm_input=64):
        super(CRNNModelTorch, self).__init__()
        assert img_h % 16 == 0, 'img_h has to be a multiple of 16'

        block_0 = BlocksTorch.conv_norm_act(n_channels, 64, pooling='max')
        block_1 = BlocksTorch.conv_norm_act(64, 128, pooling='max')
        block_2 = BlocksTorch.conv_norm_act(128, 256)
        block_3 = BlocksTorch.conv_norm_act(256, 256, pooling='max', pooling_s=(2, 1), pooling_k=(2, 1))
        block_4 = BlocksTorch.conv_norm_act(256, 512, norm="bn")
        block_5 = BlocksTorch.conv_norm_act(512, 512, pooling='max', norm="bn", pooling_s=(2, 1), pooling_k=(2, 1))
        block_6 = BlocksTorch.conv_norm_act(512, 512, k=2, p=0)
        self.cnn = nn.Sequential(block_0, block_1, block_2, block_3, block_4, block_5, block_6)
        self.cnn2rnn = nn.Linear(512 * (img_h // 16 - 1), lstm_input)

        # RNN
        self.rnn_01 = nn.LSTM(lstm_input, n_hidden, bidirectional=True)
        self.rnn_02 = nn.LSTM(2 * n_hidden, n_hidden, bidirectional=True)

        # Classifiers
        self.classifier = nn.Linear(2 * n_hidden, n_classes)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, input, return_cls=False):
        # conv features
        conv = self.cnn(input)
        batch_size, channels, h, w = conv.shape
        conv = conv.view((batch_size, channels * h, w)).permute(2, 0, 1)
        rnn_input = self.cnn2rnn(conv)
        # rnn features
        x, _ = self.rnn_01(rnn_input)
        rnn_output, _ = self.rnn_02(x)

        # Classifier
        cls = self.classifier(rnn_output)
        # add log_softmax to converge output
        output = F.log_softmax(cls, dim=2)
        if return_cls:
            return output, F.softmax(cls, 2)

        return output


if __name__ == '__main__':
    model = CRNNModelTorch(32, 1, 40, 128)
    print(model)
    sample = torch.rand(2, 1, 32, 256)
    output = model(sample)
    print(output.shape)
