import torch
import torch.nn as nn
import torch.nn.functional as f


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def up_conv2x2(in_channels, out_channels, transpose=True):
    if transpose:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class FinalConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, residual=True, batch_norm=True):
        super(FinalConv, self).__init__()
        self.residual = residual
        self.batch_norm = batch_norm
        self.bn = None
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = []
        self.conv3 = conv1x1(out_channels, out_channels)
        for _ in range(blocks - 1):
            self.conv2.append(conv3x3(out_channels, out_channels))
        if self.batch_norm:
            self.bn = []
            for _ in range(blocks):
                self.bn.append(nn.BatchNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        self.conv2 = nn.ModuleList(self.conv2)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x1 = f.relu(self.conv1(x))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            if self.batch_norm:
                x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = f.relu(x2)
            x1 = x2
        return self.conv3(x2)


class UpConvD(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, residual=True, batch_norm=True, transpose=True, concat=True):
        super(UpConvD, self).__init__()
        self.concat = concat
        self.residual = residual
        self.batch_norm = batch_norm
        self.bn = None
        self.conv2 = []
        self.up_conv = up_conv2x2(in_channels, out_channels, transpose=transpose)
        if self.concat:
            self.conv1 = conv3x3(2 * out_channels, out_channels)
        else:
            self.conv1 = conv3x3(out_channels, out_channels)
        for _ in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels))
        if self.batch_norm:
            self.bn = []
            for _ in range(blocks):
                self.bn.append(nn.BatchNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        self.conv2 = nn.ModuleList(self.conv2)

    def __call__(self, from_up, from_down):
        return self.forward(from_up, from_down)

    def forward(self, from_up, from_down):
        from_up = self.up_conv(from_up)
        if self.concat:
            x1 = torch.cat((from_up, from_down), 1)
        else:
            if from_down is not None:
                x1 = from_up + from_down
            else:
                x1 = from_up
        x1 = f.relu(self.conv1(x1))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            if self.batch_norm:
                x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = f.relu(x2)
            x1 = x2
        return x2


class DownConvD(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, pooling=True, residual=True, batch_norm=True):
        super(DownConvD, self).__init__()
        self.pooling = pooling
        self.residual = residual
        self.batch_norm = batch_norm
        self.bn = None
        self.pool = None
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels))
        if self.batch_norm:
            self.bn = []
            for _ in range(blocks):
                self.bn.append(nn.BatchNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.ModuleList(self.conv2)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x1 = f.relu(self.conv1(x))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            if self.batch_norm:
                x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = f.relu(x2)
            x1 = x2
        before_pool = x2
        if self.pooling:
            x2 = self.pool(x2)
        return x2, before_pool
