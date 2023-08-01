import torch
from torch import nn, Tensor
from .DropBlock import DropBlock
from .MAP import MAP


class Conv1(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(Conv1, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, dilation=1),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )


class Conv2(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None, num=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(Conv2, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, bias=False, dilation=2),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )


class Conv3(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(Conv3, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=3, bias=False, dilation=3),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )


class LConv3(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(LConv3, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=3, bias=False, dilation=3),
            DropBlock(7, 0.18),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )


class DoubleConv1(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv1, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, dilation=1),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, dilation=1),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DoubleConv2(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv2, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=2, bias=False, dilation=2),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=2, bias=False, dilation=2),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class DoubleConv3(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv3, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=3, bias=False, dilation=3),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=3, bias=False, dilation=3),
            DropBlock(7, 0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ResPath1(nn.Module):
    def __init__(self, in_channels):
        super(ResPath1, self).__init__()
        self.conv1 = Conv1(in_channels, in_channels)
        self.conv2 = Conv2(in_channels, in_channels)
        self.conv3 = Conv3(in_channels, in_channels)
        self.conv4 = Conv1(in_channels, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = x + x4
        return x5


class ResPath2(nn.Module):
    def __init__(self, in_channels):
        super(ResPath2, self).__init__()
        self.conv1 = Conv2(in_channels, in_channels)
        self.conv2 = Conv3(in_channels, in_channels)
        self.conv3 = Conv1(in_channels, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = x + x3
        return x4


class ResPath3(nn.Module):
    def __init__(self, in_channels):
        super(ResPath3, self).__init__()
        self.conv1 = Conv3(in_channels, in_channels)
        self.conv2 = Conv1(in_channels, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = x + x2
        return x3


class ResPath4(nn.Module):
    def __init__(self, in_channels):
        super(ResPath4, self).__init__()
        self.conv1 = Conv1(in_channels, in_channels)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x2 = x + x1
        return x2


class MRIConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels // 3
        super(MRIConv, self).__init__()
        self.branch1 = nn.Sequential(
            Conv1(in_channels, out_channels),
            nn.Conv2d(out_channels, mid_channels, kernel_size=1)
        )
        self.branch2 = nn.Sequential(
            Conv2(in_channels, out_channels),
            nn.Conv2d(out_channels, mid_channels, kernel_size=1)
        )
        self.branch3 = nn.Sequential(
            Conv3(in_channels, out_channels),
            nn.Conv2d(out_channels, mid_channels, kernel_size=1)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x5 = self.branch4(x)
        x5 = x4 + x5
        return x5


class MROConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None, scale_factor=None):
        if mid_channels is None:
            mid_channels = in_channels // 6
        super(MROConv, self).__init__()
        self.branch1 = nn.Sequential(
            Conv1(in_channels, 3 * mid_channels),
            nn.Conv2d(3 * mid_channels, mid_channels, kernel_size=1)
        )
        self.branch2 = nn.Sequential(
            Conv2(in_channels, 3 * mid_channels),
            nn.Conv2d(3 * mid_channels, mid_channels, kernel_size=1)
        )
        self.branch3 = nn.Sequential(
            Conv3(in_channels, 3 * mid_channels),
            nn.Conv2d(3 * mid_channels, mid_channels, kernel_size=1)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, 3 * mid_channels, kernel_size=1)
        )
        self.Conv5 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = torch.cat([x1, x2, x3], dim=1)
        x = self.branch4(x)
        x5 = x4 + x
        x6 = self.Conv5(x5)
        x7 = self.up(x6)
        return x7


class Down1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down1, self).__init__()
        self.DownSample = nn.MaxPool2d(2, stride=2)
        self.Conv = DoubleConv1(2 * in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x2 = self.DownSample(x2)
        x3 = torch.cat([x1, x2], dim=1)
        x4 = self.Conv(x3)

        return x4


class Down2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down2, self).__init__()
        self.DownSample = nn.MaxPool2d(2, stride=2)
        self.Conv = DoubleConv2(2 * in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x2 = self.DownSample(x2)
        x3 = torch.cat([x1, x2], dim=1)
        x4 = self.Conv(x3)

        return x4


class Down3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down3, self).__init__()
        self.DownSample = nn.MaxPool2d(2, stride=2)
        self.Conv = DoubleConv3(2 * in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x2 = self.DownSample(x2)
        x3 = torch.cat([x1, x2], dim=1)
        x4 = self.Conv(x3)

        return x4


class Down4(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = in_channels // 2
        super(Down4, self).__init__()
        self.MP = nn.MaxPool2d(2, stride=2)
        self.Conv1 = LConv3(mid_channels, in_channels)
        self.PPM = MAP(in_channels, [2, 3, 5, 7])
        self.Conv2 = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1)
        self.Conv3 = LConv3(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.MP(x)
        x2 = self.Conv1(x1)
        x3 = self.PPM(x2)
        x4 = self.Conv2(x3)
        x5 = self.Conv3(x4)

        return x5


class Up1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up1, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x2 = self.up(x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Up2(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up2, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv1(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv1(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x2 = self.up(x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Up3(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up3, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv3(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x2 = self.up(x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, bias=False, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.Sigmoid(),
        )


class MDR_Net(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 bilinear: bool = False,
                 base_c: int = 36):
        super(MDR_Net, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.MRI1 = MRIConv(in_channels, base_c // 2)
        self.down1 = DoubleConv1(base_c // 2, base_c)
        self.ResPath1 = ResPath1(base_c)

        self.DownSample = nn.MaxPool2d(2, stride=2)

        self.MRI2 = MRIConv(in_channels, base_c)
        self.down2 = Down1(base_c, base_c * 2)
        self.ResPath2 = ResPath2(base_c * 2)

        self.MRI3 = MRIConv(in_channels, base_c * 2)
        self.down3 = Down2(base_c * 2, base_c * 4)
        self.ResPath3 = ResPath3(base_c * 4)

        self.MRI4 = MRIConv(in_channels, base_c * 4)
        self.down4 = Down3(base_c * 4, base_c * 8)
        self.ResPath4 = ResPath4(base_c * 8)

        factor = 2 if bilinear else 1
        self.down5 = Down4(base_c * 16, base_c * 16 // factor)

        self.up1 = Up1(base_c * 16, base_c * 8 // factor, bilinear)
        self.MRO1 = MROConv(base_c * 8, num_classes, scale_factor=8)

        self.up2 = Up2(base_c * 8, base_c * 4 // factor, bilinear)
        self.MRO2 = MROConv(base_c * 4, num_classes, scale_factor=4)

        self.up3 = Up2(base_c * 4, base_c * 2 // factor, bilinear)
        self.MRO3 = MROConv(base_c * 2, num_classes, scale_factor=2)

        self.up4 = Up3(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)
        self.out_conv1 = OutConv(4 * num_classes, num_classes)

    def forward(self, x):
        x1 = self.MRI1(x)
        x2 = self.down1(x1)
        x3 = self.ResPath1(x2)

        x4 = self.DownSample(x)
        x5 = self.MRI2(x4)
        x6 = self.down2(x5, x2)
        x7 = self.ResPath2(x6)

        x8 = self.DownSample(x4)
        x9 = self.MRI3(x8)
        x10 = self.down3(x9, x6)
        x11 = self.ResPath3(x10)

        x12 = self.DownSample(x8)
        x13 = self.MRI4(x12)
        x14 = self.down4(x13, x10)
        x15 = self.ResPath4(x14)

        x16 = self.down5(x14)

        x17 = self.up1(x15, x16)
        x18 = self.MRO1(x17)

        x19 = self.up2(x11, x17)
        x20 = self.MRO2(x19)

        x21 = self.up3(x7, x19)
        x22 = self.MRO3(x21)

        x23 = self.up4(x3, x21)
        x24 = self.out_conv(x23)
        outputs = [0.2 * x18, 0.4 * x20, 0.6 * x22, x24]
        x25 = torch.cat(outputs, dim=1)
        logits = self.out_conv1(x25)

        return logits


if __name__ == "__main__":
    model = MDR_Net()
