from __future__ import print_function, division
from configparser import BasicInterpolation
from functools import partial
from numpy.core.multiarray import concatenate

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

norm_op_kwargs = {'eps': 1e-5, 'affine': True}
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}
nonlinearity = partial(F.relu, inplace=True)

class MRDPNet(nn.Module):
    """HMRNet-p is designed for plump anatomical structures
             which contains 3x3x3 convolutional blocks."""

    def __init__(self, params):
        super(MRDPNet, self).__init__()
        self.params = params
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.fr_chs = self.params['fr_feature_chns']
        self.num_classes = self.params['class_num']
        self.bilinear = self.params['bilinear']
        self.conv_op = self.params['con_op']
        self._deep_supervision = self.params['_deep_supervision']
        self.do_ds = self.params['do_ds']

        filters = [32, 64, 128, 256]

        assert (len(self.ft_chns) == 5 or len(self.ft_chns) == 4)

        self.in_conv = ConvBlock(self.in_chns, self.ft_chns[0])
        self.fr_in_conv = ConvBlock(self.in_chns, self.fr_chs)

        self.SK_e1 = ACIM(filters[0], filters[0], 0)
        self.SK_e11 = ACIM(filters[0] // 4, filters[0] // 4, 1)
        self.SK_e2 = ACIM(filters[1], filters[1], 0)
        self.SK_e22 = ACIM(filters[0] // 2, filters[0] // 4, 1)
        self.SK_e3 = ACIM(filters[2], filters[2], 0)
        self.SK_e33 = ACIM(filters[0], filters[0] // 4, 1)
        self.SK_d1 = ACIM(filters[1], filters[1], 0)
        self.SK_d11 = ACIM(filters[0] // 2, filters[0] // 4, 1)

        self.SK_e4 = ACIM(filters[3], filters[0])
        self.SK_d3 = ACIM(filters[0] // 2, filters[0]// 2, 0)
        self.SK_d2 = ACIM(filters[0], filters[0],0)



        self.basic_conv1 = DSHAM(self.fr_chs)  # full _resolution conv
        self.down1 = CAEM(self.ft_chns[0], self.ft_chns[1])  # Unet  feature down sample
        self.basic_down1 = DownBlock_Con(pooling_p=2)  # full_resolution down sample
        self.basic_up1 = UpBlock_Con(scale_factor=2)  # Unet  feature up sample
        self.concat1 = Concatenate(self.ft_chns[1] + self.fr_chs, self.ft_chns[1])
        self.fr_concat1 = Concatenate(self.ft_chns[1] + self.fr_chs, self.fr_chs)

        self.basic_conv2 = DSHAM(self.fr_chs)
        self.down2 = CAEM(self.ft_chns[1], self.ft_chns[2])
        self.basic_down2 = DownBlock_Con(pooling_p=4)
        self.basic_up2 = UpBlock_Con(scale_factor=4)
        self.concat2 = Concatenate(self.ft_chns[2] + self.fr_chs, self.ft_chns[2])
        self.fr_concat2 = Concatenate(self.ft_chns[2] + self.fr_chs, self.fr_chs)

        self.basic_conv3 = DSHAM(self.fr_chs)
        self.down3 = CAEM(self.ft_chns[2], self.ft_chns[3])
        self.basic_down3 = DownBlock_Con(pooling_p=8)
        self.basic_up3 = UpBlock_Con(scale_factor=8)
        self.dpvision_con3 = nn.Conv2d(self.ft_chns[3], self.num_classes, kernel_size=1)  # deep supervison
        self.concat3 = Concatenate(self.ft_chns[3] + self.fr_chs, self.ft_chns[3])
        self.fr_concat3 = Concatenate(self.ft_chns[3] + self.fr_chs, self.fr_chs)

        if (len(self.ft_chns) == 5):
            self.basic_conv4 = DSHAM(self.fr_chs)
            self.down4 = CAEM(self.ft_chns[3], self.ft_chns[4])
            self.basic_down4 = DownBlock_Con(self.fr_chs, self.ft_chns[4], pooling_p=16)
            self.basic_up4 = UpBlock_Con(self.ft_chns[4], self.fr_chs, scale_factor=16)
            self.concat4 = Concatenate(self.ft_chns[4])
            self.fr_concat4 = Concatenate(self.fr_chs)

            self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3],
                               bilinear=self.bilinear)

        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2],
                           bilinear=self.bilinear)
        self.basic_conv5 = DSHAM(self.fr_chs)
        self.basic_down5 = DownBlock_Con(pooling_p=4)
        self.basic_up5 = UpBlock_Con(scale_factor=4)
        self.dpvision_con5 = nn.Conv2d(self.ft_chns[2], self.num_classes, kernel_size=1)  # deep supervison
        self.concat5 = Concatenate_Threechs(2 * self.ft_chns[2] + self.fr_chs, self.ft_chns[2])
        self.fr_concat5 = Concatenate(self.ft_chns[2] + self.fr_chs, self.fr_chs)

        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1],
                           bilinear=self.bilinear)
        self.basic_conv6 = DSHAM(self.fr_chs)
        self.basic_down6 = DownBlock_Con(pooling_p=2)
        self.basic_up6 = UpBlock_Con(scale_factor=2)
        self.dpvision_con6 = nn.Conv2d(self.ft_chns[1], self.num_classes, kernel_size=1)  # deep supervison
        self.concat6 = Concatenate_Threechs(2 * self.ft_chns[1] + self.fr_chs, self.ft_chns[1])
        self.fr_concat6 = Concatenate(self.ft_chns[1] + self.fr_chs, self.fr_chs)

        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0],
                           bilinear=self.bilinear)
        self.basic_conv7 = DSHAM(self.fr_chs)
        self.concat7 = Concatenate_Threechs(2 * self.ft_chns[0] + self.fr_chs, self.ft_chns[0])
        self.fr_concat7 = Concatenate(self.ft_chns[0] + self.fr_chs, self.fr_chs)
        self.final_concat = Concatenate(self.ft_chns[0] + self.fr_chs, self.ft_chns[0])

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.num_classes,
                                  kernel_size=3, padding=1)
        self.softmax = lambda x: F.softmax(x, 1)

    def forward(self, x):
        segout = []

        x0 = self.in_conv(x)
        fr_x0 = self.fr_in_conv(x)

        x1 = self.down1(x0)  # conv  + down sample
        x11 = self.basic_up1(x1)  # conv  + up sample
        fr_x1 = self.basic_conv1(fr_x0)  # conv
        fr_x11 = self.basic_down1(fr_x1)  # conv  + down sample
        x1 = self.SK_e1(x1, fr_x11)
        fr_x1 = self.SK_e11(x11, fr_x1)
        x2 = self.down2(x1)
        x22 = self.basic_up2(x2)
        fr_x2 = self.basic_conv2(fr_x1)
        fr_x22 = self.basic_down2(fr_x2)
        x2 = self.concat2(x2, fr_x22)
        x2 = self.SK_e2(x2, fr_x22)
        fr_x2 = self.fr_concat2(fr_x2, x22)
        fr_x2 = self.SK_e22(x22, fr_x2)

        x3 = self.down3(x2)
        x33 = self.basic_up3(x3)
        fr_x3 = self.basic_conv3(fr_x2)
        fr_x33 = self.basic_down3(fr_x3)
        x3 = self.concat3(x3, fr_x33)
        x3 = self.SK_e3(x3, fr_x33)
        deep_x3 = self.dpvision_con3(x3)
        segout.append(deep_x3)
        fr_x3 = self.fr_concat3(fr_x3, x33)
        fr_x3 = self.SK_e33(x33, fr_x3)

        if (len(self.ft_chns) == 5):
            x4 = self.down4(x3)
            x44 = self.basic_up4(x4)
            fr_x4 = self.basic_conv4(fr_x3)
            fr_x44 = self.basic_down4(fr_x4)
            x4 = self.concat4(x4, fr_x44)
            fr_x4 = self.fr_concat4(fr_x4, x44)

            x = self.up1(x4, x3)

        else:
            x = x3
            fr_x = fr_x3

        x5 = self.up2(x)
        x55 = self.basic_up5(x5)
        fr_x5 = self.basic_conv5(fr_x)
        fr_x55 = self.basic_down5(fr_x5)
        x5 = self.concat5(x5, fr_x55, x2)
        x5 = self.SK_d1(x5, fr_x55)
        deep_x5 = self.dpvision_con5(x5)
        segout.append(deep_x5)
        fr_x5 = self.fr_concat5(fr_x5, x55)
        fr_x5 = self.SK_d11(x55, fr_x5)

        x6 = self.up3(x5)
        x66 = self.basic_up6(x6)
        fr_x6 = self.basic_conv6(fr_x5)
        fr_x66 = self.basic_down6(fr_x6)
        x6 = self.concat6(x6, fr_x66, x1)
        x6 = self.SK_d2(x6,fr_x66)
        deep_x6 = self.dpvision_con6(x6)
        segout.append(deep_x6)
        fr_x6 = self.fr_concat6(fr_x6, x66)
        fr_x6 = self.SK_e11(x66, fr_x6)

        x7 = self.up4(x6)
        x7_ = x7
        fr_x7 = self.basic_conv7(fr_x6)
        x7 = self.SK_d3(x7, fr_x7)

        x7 = self.final_concat(x7, fr_x7)

        output = self.out_conv(x7)

        return output

    def name(self):
        return "MRDPNet"


class ACIM(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, p=0):
        super(ACIM, self).__init__()
        self.p = p
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, in_channels, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels * 4, 8, kernel_size=1, padding=0)
        self.sa = CoordAtt(in_channels, in_channels)
        self.sa1 = CoordAtt(8, 8)
        self.csa = eca_layer(out_channels * 2)

    def forward(self, x1, x2):
        # SAT路径
        if self.p == 0:
            x2 = self.conv2(x2)
            sa_out1 = self.sa(x1)
            sa_out2 = self.sa(x2)
        if self.p == 1:
            x1 = self.conv3(x1)
            sa_out1 = self.sa1(x1)
            sa_out2 = self.sa1(x2)
        f1 = sa_out1 * x2
        f2 = sa_out2 * x1
        f1 = f1 + x1
        f2 = f2 + x2
        f = torch.cat([f1, f2], dim=1)

        csa_out = self.csa(f)
        csa_out =self.conv1(csa_out)

        return csa_out


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
        source: https://github.com/BangguWu/ECANet
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels):
        """
: probability to be zeroed
        """
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, **norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, **norm_op_kwargs),
            nn.LeakyReLU(**net_nonlin_kwargs)
        )

    def forward(self, x):
        x = self.conv_conv(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # 深度卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batch_norm1 = nn.InstanceNorm2d(in_channels)
        self.batch_norm2 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # 深度卷积
        x = self.batch_norm1(self.depthwise(x))
        x = self.relu(x)
        # 逐点卷积
        x = self.pointwise(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        return x

class ACA(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ACA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)

class DSHAM(nn.Module):
    def __init__(self, input_chns):
        super(DSHAM, self).__init__()

        self.depthwise1 = DepthwiseSeparableConv(input_chns, input_chns)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.ACA = ACA(input_chns, reduction_ratio=8)
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.conv1 = nn.Conv2d(input_chns, input_chns, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 获取输入的形状
        batch_size, channels, height, width = x.size()

        x_max = self.maxpool(x)

        # 计算每个部分的高度和宽度
        h_half = height // 2
        w_half = width // 2

        # 将 x 在 H 和 W 维度上分成四份
        x1 = x[:, :, :h_half, :w_half]  # 左上部分
        x2 = x[:, :, :h_half, w_half:]  # 右上部分
        x3 = x[:, :, h_half:, :w_half]  # 左下部分
        x4 = x[:, :, h_half:, w_half:]  # 右下部分
        x1 = self.depthwise1(x1)
        x2 = self.depthwise1(x2)
        x3 = self.depthwise1(x3)
        x4 = self.depthwise1(x4)
        x1 = x1 * x_max
        x2 = x2 * x_max
        x3 = x3 * x_max
        x4 = x4 * x_max

        x1 = self.ACA(x1)
        x2 = self.ACA(x2)
        x3 = self.ACA(x3)
        x4 = self.ACA(x4)
        # 将 x1, x2, x3, x4 按照原位置拼回去
        top = torch.cat((x1, x2), dim=3)  # 在宽度维度拼接
        bottom = torch.cat((x3, x4), dim=3)  # 在宽度维度拼接
        output = torch.cat((top, bottom), dim=2)  # 在高度维度拼接

        max_pool = torch.max(output, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(output, dim=1, keepdim=True)

        concat = torch.cat([max_pool, avg_pool], dim=1)
        sa_map = torch.sigmoid(self.conv(concat))
        output = x * sa_map
        output = torch.sigmoid(output)
        output = output * x
        return output

class CAEM(nn.Module):
    """Downsampling before ConvBlock"""

    def __init__(self, in_channels, out_channels):
        super(CAEM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        y = self.conv1(x)
        y = self.max_pool(y)
        z = torch.mul(x, y)
        x = self.maxpool_conv(z)
        return x

class DownBlock_Con(nn.Module):
    """Downsampling """

    def __init__(self, pooling_p):
        super(DownBlock_Con, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pooling_p),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock_Con(nn.Module):
    """Upampling """

    def __init__(self, scale_factor=2):
        super(UpBlock_Con, self).__init__()
        self.uppool_conv = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        return self.uppool_conv(x)


class Concatenate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Concatenate, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Concatenate_Threechs(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Concatenate_Threechs, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2, x3):
        x = torch.cat([x3, x2, x1], dim=1)
        return self.conv(x)


class UpBlock(nn.Module):
    """Upssampling before ConvBlock"""

    def __init__(self, in_channels, out_channels,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.uppool_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                ConvBlock(in_channels, out_channels)
            )

    def forward(self, x):
        x = self.uppool_conv(x)
        return x

class CoordAtt(nn.Module):

    class h_sigmoid(nn.Module):
        def __init__(self, inplace=True):
            super(CoordAtt.h_sigmoid, self).__init__()
            self.relu = nn.ReLU6(inplace=inplace)

        def forward(self, x):
            return self.relu(x + 3) / 6

    class h_swish(nn.Module):
        def __init__(self, inplace=True):
            super(CoordAtt.h_swish, self).__init__()
            self.sigmoid = CoordAtt.h_sigmoid(inplace=inplace)

        def forward(self, x):
            return x * self.sigmoid(x)

    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = self.h_swish()          # 使用内部类

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out

if __name__ == "__main__":
    from thop import profile

    params = {'in_chns': 3,
              'class_num': 1,
              'feature_chns': [16, 32, 64, 128],
              'fr_feature_chns': 8,
              'bilinear': True,
              '_deep_supervision': True,
              'do_ds': True,
              'con_op': True}
    models = MRDPNet(params).cuda()
    models.eval()
    x = torch.rand([4, 3, 256, 256]).cuda(0)
    out = models(x)
    print(out.shape)
    total = sum([param.nelement() for param in models.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    flops, params = profile(models, inputs=(x,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
