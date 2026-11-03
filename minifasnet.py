"""
MiniFASNetV2 - Silent Face Anti-Spoofing Model
Kiến trúc khớp với file 2.7_80x80_MiniFASNetV2.pth
"""
import torch
import torch.nn as nn

class ConvBn(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x): return self.bn(self.conv(x))

class ConvBnPrelu(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=1, padding=0, groups=1):
        super(ConvBnPrelu, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x): return self.prelu(self.bn(self.conv(x)))

class MiniFASNetBlock(nn.Module):
    def __init__(self, in_c, out_c, expansion, kernel=(3, 3), stride=1, padding=1, residual=True):
        super(MiniFASNetBlock, self).__init__()
        self.residual = residual and (in_c == out_c) and (stride == 1)
        self.conv = ConvBnPrelu(in_c, expansion, kernel=(1, 1), padding=0, stride=1)
        self.conv_dw = ConvBnPrelu(expansion, expansion, kernel=kernel, padding=padding, stride=stride, groups=expansion)
        self.project = ConvBn(expansion, out_c, kernel=(1, 1), padding=0, stride=1)

    def forward(self, x):
        out = self.project(self.conv_dw(self.conv(x)))
        if self.residual:
            return x + out
        return out

class MiniFASNetV2(nn.Module):
    def __init__(self, embedding_size=128, num_classes=3):
        super(MiniFASNetV2, self).__init__()
        
        # 80x80 -> 40x40
        self.conv1 = ConvBnPrelu(3, 32, kernel=(3, 3), stride=2, padding=1)
        self.conv2_dw = ConvBnPrelu(32, 32, kernel=(3, 3), stride=1, padding=1, groups=32)
        
        # 40x40 -> 20x20
        self.conv_23 = MiniFASNetBlock(32, 64, expansion=103, kernel=(3, 3), stride=2, padding=1, residual=False)
        
        # 20x20 -> 20x20
        self.conv_3 = nn.Sequential(
            MiniFASNetBlock(64, 64, expansion=13, kernel=(3, 3), stride=1, padding=1, residual=True),
            MiniFASNetBlock(64, 64, expansion=13, kernel=(3, 3), stride=1, padding=1, residual=True),
            MiniFASNetBlock(64, 64, expansion=13, kernel=(3, 3), stride=1, padding=1, residual=True),
            MiniFASNetBlock(64, 64, expansion=13, kernel=(3, 3), stride=1, padding=1, residual=True)
        )
        
        # 20x20 -> 10x10
        self.conv_34 = MiniFASNetBlock(64, 128, expansion=231, kernel=(3, 3), stride=2, padding=1, residual=False)
        
        # 10x10 -> 10x10
        self.conv_4 = nn.Sequential(
            MiniFASNetBlock(128, 128, expansion=231, kernel=(3, 3), stride=1, padding=1, residual=True),
            MiniFASNetBlock(128, 128, expansion=52,  kernel=(3, 3), stride=1, padding=1, residual=True),
            MiniFASNetBlock(128, 128, expansion=26,  kernel=(3, 3), stride=1, padding=1, residual=True),
            MiniFASNetBlock(128, 128, expansion=77,  kernel=(3, 3), stride=1, padding=1, residual=True),
            MiniFASNetBlock(128, 128, expansion=26,  kernel=(3, 3), stride=1, padding=1, residual=True),
            MiniFASNetBlock(128, 128, expansion=26,  kernel=(3, 3), stride=1, padding=1, residual=True)
        )
        
        # 10x10 -> 5x5
        self.conv_45 = MiniFASNetBlock(128, 128, expansion=308, kernel=(3, 3), stride=2, padding=1, residual=False)
        
        # 5x5 -> 5x5
        self.conv_5 = nn.Sequential(
            MiniFASNetBlock(128, 128, expansion=26, kernel=(3, 3), stride=1, padding=1, residual=True),
            MiniFASNetBlock(128, 128, expansion=26, kernel=(3, 3), stride=1, padding=1, residual=True)
        )
        
        # 5x5 -> 1x1
        self.conv_6_sep = ConvBnPrelu(128, 512, kernel=(1, 1), stride=1, padding=0)
        self.conv_6_dw = ConvBn(512, 512, kernel=(5, 5), stride=1, padding=0, groups=512)
        
        # Classifier
        self.linear = nn.Linear(512, embedding_size, bias=False)
        self.bn = nn.BatchNorm1d(embedding_size)
        self.prob = nn.Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        x = self.conv_6_sep(x)
        x = self.conv_6_dw(x)
        x = x.view(x.size(0), -1)
        x = self.bn(self.linear(x))
        return self.prob(x)
