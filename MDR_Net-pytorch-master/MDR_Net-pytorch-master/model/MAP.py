import torch
import torch.nn.functional as F
from torch import nn


class MAP(nn.Module):
    def __init__(self, in_dim, bins):
        super(MAP, self).__init__()
        branch_channels = in_dim // 4
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),  # nn.AdaptiveMaxPool2d(bin),
                nn.Conv2d(in_dim, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            temp = f(x)
            temp = F.interpolate(temp, x_size[2:], mode="bilinear", align_corners=True)
            out.append(temp)

        return torch.cat(out, 1)
