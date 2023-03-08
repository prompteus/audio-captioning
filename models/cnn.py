import torch.nn as nn
import torch.nn.functional as F


def calc_padding(kernel):
    try:
        return kernel // 3
    except TypeError:
        return [k // 3 for k in kernel]


class BasicBlock(nn.Module):
    """
    The basic convolutional residual block our network uses.
    """
    def __init__(self, in_channels, out_channels, stride=(1, 1), k1=3, k2=3):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(k1, k1),
            stride=stride,
            padding=calc_padding(k1),
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(k2, k2),
            stride=(1, 1),
            padding=calc_padding(k2),
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # the following is for the residual connection
        # if our block has in_channels != out_channels, we need to use a pointwise (1 x 1 kernel) convolution to
        # fix the sizes before the addition
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module(
                'conv',
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=stride,  # downsample
                    padding=0,
                    bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = F.relu(y, inplace=True)  # apply ReLU after addition
        return y


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        input_shape = config['input_shape']
        n_classes = config['n_classes']
        channels_multiplier = config['channels_multiplier']
        base_channels = config['base_channels']

        n_channels = [
            base_channels,
            base_channels * channels_multiplier,
            base_channels * channels_multiplier,
            base_channels * channels_multiplier * channels_multiplier,
            base_channels * channels_multiplier * channels_multiplier * channels_multiplier
        ]

        # for this simple baseline model, we build:
        #  - an input convolutional with a large kernel (5) and a stride of (2) to downsample the input and
        #    therefore save a lot of compute
        #  - 8 convolutional blocks consisting of:
        #    conv -> bn -> relu -> conv -> bn -> relu; including a residual connection before the last relu activation
        #  - a convolutional classification head (fully-convolutional model) with as many output channels as
        #    classes to predict

        self.in_c = nn.Sequential(nn.Conv2d(
            input_shape[1],
            n_channels[0],
            kernel_size=(5, 5),
            stride=(2, 2),
            padding=1,
            bias=False),
            nn.BatchNorm2d(n_channels[0]),
            nn.ReLU(True)
        )

        self.features = nn.Sequential(
            BasicBlock(n_channels[0], n_channels[1]),
            BasicBlock(n_channels[1], n_channels[1]),
            nn.MaxPool2d((2, 2)),
            BasicBlock(n_channels[1], n_channels[2]),
            BasicBlock(n_channels[2], n_channels[2]),
            nn.MaxPool2d((2, 2)),
            BasicBlock(n_channels[2], n_channels[3]),
            BasicBlock(n_channels[3], n_channels[3]),
            BasicBlock(n_channels[3], n_channels[4]),
            BasicBlock(n_channels[4], n_channels[4]),
        )

        ff_list = []
        ff_list += [nn.Conv2d(
            n_channels[4],
            n_classes,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=False),
            nn.BatchNorm2d(n_classes),
        ]
        # this pooling operation results in 1 value per channel -> the logit predicted for one of the classes
        ff_list.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.feed_forward = nn.Sequential(
            *ff_list
        )

    def forward(self, x):
        # batch size x channels x mel bins x time frames
        # 12 x 1 x 128 x 100
        x = self.in_c(x)
        # 12 x 16 x 63 x 49
        x = self.features(x)
        # 12 x 128 x 15 x 12
        x = self.feed_forward(x)
        # 12 x 10 x 1 x 1
        logit = x.squeeze(2).squeeze(2)
        # 12 x 10 (nr. of channels = nr. of classes)
        return logit


def get_model(in_channels=1, n_classes=10, base_channels=32, channels_multiplier=2):
    """

    :param in_channels: 1 for spectrograms, 3 for RBG images
    :param n_classes: number of logits
    :param base_channels: scales width (number of channels) of layers
    :param channels_multiplier: multiplier to control expansion ratio of layer widths
    :return: the pytorch model
    """
    model_config = {
        "base_channels": base_channels,
        "channels_multiplier": channels_multiplier,
        "input_shape": [
            10,
            in_channels,
            -1,
            -1
        ],
        "n_classes": n_classes,
    }
    m = Network(model_config)
    print(m)
    return m
