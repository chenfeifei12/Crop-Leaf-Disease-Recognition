from typing import List, Callable

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.parameter import Parameter
class Inception(nn.Module):
    def __init__(self, in_channels, out_channels1reduce):
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1reduce, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels1reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels1reduce, out_channels1reduce, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels1reduce),
            nn.ReLU(inplace=True),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1reduce, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels1reduce),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels, out_channels1reduce, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels1reduce),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out
class InceptionV3(nn.Module):
    def __init__(self, in_channels, out_channels1reduce):
        super(InceptionV3, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1reduce, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels1reduce),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out4 = self.branch4(x)
        out = torch.cat([out1, out4], dim=1)
        return out
class ECA(nn.Module):

    def __init__(self):
        super(ECA, self).__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        b, c, _, _ = x.size()
        y = x.mean((2, 3), keepdim=True)
        y1 = x.std((2, 3), keepdim=True)
        y3 = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y2 = self.conv(y1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y=y2+y3
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
def ConvBNReLU(in_channels,out_channels,kernel_size,stride=1,groups=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,groups=groups, stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels)
    )
class InceptionV3ModuleE(nn.Module):
    def __init__(self, in_channels, out_channels1reduce,groups=1,stride=2):
        super(InceptionV3ModuleE, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels1reduce, kernel_size=3, stride=stride,padding=1,groups=groups),
        )

        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels=in_channels, out_channels=out_channels1reduce, kernel_size=5, stride=stride, padding=2, groups=groups),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat([out1, out2], dim=1)
        return out
def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, input_c: int, output_c: int, stride: int):
        super(InvertedResidual, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2
        # 当stride为1时，input_channel应该是branch_features的两倍
        # python中 '<<' 是位运算，可理解为计算×2的快速方法
        assert (self.stride != 1) or (input_c == branch_features << 1)

        if self.stride == 2:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                nn.Conv2d(input_c, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                ECA()
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(input_c if self.stride > 1 else branch_features, branch_features, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            InceptionV3ModuleE(branch_features, branch_features,groups=branch_features, stride=self.stride),
            nn.Conv2d(2*branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            ECA()
        )
        self.branch3 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                     nn.Conv2d(input_c, output_c, kernel_size=1, stride=1, padding=0, bias=False),
                                     nn.BatchNorm2d(output_c),
                                     nn.ReLU(inplace=True))

    @staticmethod
    def depthwise_conv(input_c: int,
                       output_c: int,
                       kernel_s: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(in_channels=input_c, out_channels=output_c, kernel_size=kernel_s,
                         stride=stride, padding=padding, bias=bias, groups=input_c)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            x3 =x2+self.branch2(x2)
            out = torch.cat((x1, x3), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
            out = out + self.branch3(x)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self,
                 stages_repeats: List[int],
                 stages_out_channels: List[int],
                 num_classes: int = 1000,
                 inverted_residual: Callable[..., nn.Module] = InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")
        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")
        self._stage_out_channels = stages_out_channels

        # input RGB image
        input_channels = 3
        output_channels = self._stage_out_channels[0]

        self.conv1 = nn.Sequential(
            Inception(input_channels, output_channels)
        )

        self.maxpool = InceptionV3(72, 72)

        input_channels = 6*output_channels

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(stage_names, stages_repeats,
                                                  self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(output_channels, num_classes)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # global pool
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def shufflenet_v2_x0_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[3, 4, 3],
                         stages_out_channels=[24, 48, 96, 192, 512],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x1_0(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[3, 4, 3],
                         stages_out_channels=[24, 96, 192, 384, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x1_5(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x1_5-3c479a10.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 176, 352, 704, 1024],
                         num_classes=num_classes)

    return model


def shufflenet_v2_x2_0(num_classes=1000):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`.
    weight: https://download.pytorch.org/models/shufflenetv2_x2_0-8be3c8ee.pth

    :param num_classes:
    :return:
    """
    model = ShuffleNetV2(stages_repeats=[4, 8, 4],
                         stages_out_channels=[24, 244, 488, 976, 2048],
                         num_classes=num_classes)

    return model
