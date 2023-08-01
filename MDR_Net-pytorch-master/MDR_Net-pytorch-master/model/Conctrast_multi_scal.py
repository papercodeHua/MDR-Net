import torch
from torch import nn, Tensor


# OFU
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, dilation=1),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, dilation=1),
        )


class OFU(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(OFU, self).__init__()
        # self.DownSample = nn.MaxPool2d(2, stride=2)
        self.conv1 = DoubleConv(in_channels, mid_channels)
        self.conv2 = DoubleConv(in_channels, mid_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> torch.Tensor:
        x3 = self.conv1(x1)
        x4 = torch.cat([x3, x2], dim=1)
        x5 = self.conv2(x4)
        return x5


# MLI
class MLI(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):  # 1 32
        if mid_channels is None:
            mid_channels = out_channels // 2  # 16
        super(MLI, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, dilation=1),
            nn.Conv2d(mid_channels, mid_channels // 8, kernel_size=1)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, bias=False, dilation=2),
            nn.Conv2d(mid_channels, mid_channels // 8, kernel_size=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=3, bias=False, dilation=3),
            nn.Conv2d(mid_channels, mid_channels // 8, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch1(x)  # 2
        x2 = self.branch2(x)  # 2
        x3 = self.branch3(x)  # 2
        x4 = torch.cat([x1, x2, x3], dim=1)  # 6
        return x4


# FAF

class FAF(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(FAF, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.Relu = nn.ReLU(inplace=True)

    def forward(self, x1: Tensor, x2: Tensor) -> torch.Tensor:
        x3 = torch.cat([x1, x2], dim=1)
        x4 = self.conv1(x3)
        x5 = self.Relu(x4)
        x6 = self.conv2(x5)
        x7 = self.Relu(x6)
        return x7


# RCU

class RCU(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(RCU, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, dilation=1)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, dilation=1)
        self.Relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> torch.Tensor:
        x2 = self.Relu(x)
        x3 = self.conv1(x2)
        x4 = self.Relu(x3)
        x5 = self.conv2(x4)
        x6 = x + x5
        return x6
