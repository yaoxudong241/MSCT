from model import common
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from util.non import NONLocalBlock2D
from util.tools import extract_image_patches, \
    reduce_mean, reduce_sum, same_padding, reverse_patches
from torchsummary import summary
from util.transformer import drop_path, DropPath, PatchEmbed, Mlp, MLABlock, CMTBlock,EfficientViTBlockyuanshi
from util.position import PositionEmbeddingLearned, PositionEmbeddingSine
import pdb
import math


def make_model(upscale=4):
    # inpu = torch.randn(1, 3, 320, 180).cpu()
    # flops, params = profile(RTC(upscale).cpu(), inputs=(inpu,))
    # print(params)
    # print(flops)
    return myCmt(upscale=upscale)

def complex_relu(input):
    return F.relu(input.real).type(torch.complex64)+1j* F.relu(input.imag).type(torch.complex64)

def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return apply_complex(self.conv_r, self.conv_i, input)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class one_conv(nn.Module):
    def __init__(self, inchanels, growth_rate, kernel_size=3, relu=True):
        super(one_conv, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.conv = nn.Conv2d(inchanels, growth_rate, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        self.flag = relu  # 是否采用激活函数
        self.conv1 = nn.Conv2d(growth_rate, inchanels, kernel_size=kernel_size, padding=kernel_size >> 1, stride=1)
        if relu:
            self.relu = nn.PReLU(growth_rate)
        self.weight1 = common.Scale(1)
        self.weight2 = common.Scale(1)  # 两个weight用于调整Residual Unit中输入和输出的权值

    def forward(self, x):
        if self.flag == False:
            output = self.weight1(x) + self.weight2(self.conv1(self.conv(x)))
        else:
            output = self.weight1(x) + self.weight2(
                self.conv1(self.relu(self.conv(x))))  # 残差模块，对应文中arfb模块中的Residual Unit
        return output  # torch.cat((x,output),1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, relu=True,
                 bn=False, bias=False, up_size=0, fan=False):
        super(BasicConv, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.out_channels = out_planes
        self.in_channels = in_planes
        if fan:
            self.conv = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                           padding=padding,
                                           dilation=dilation, groups=groups, bias=bias)
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size
        self.up_sample = nn.Upsample(size=(up_size, up_size), mode='bilinear') if up_size != 0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = self.up_sample(x)
        return x


class one_module(nn.Module):  # 整个arfb模块
    def __init__(self, n_feats):
        super(one_module, self).__init__()
        self.layer1 = one_conv(n_feats, n_feats // 2, 3)  # 图6中第一个ru
        self.layer2 = one_conv(n_feats, n_feats // 2, 3)  # 图6中第二个ru
        # self.layer3 = one_conv(n_feats, n_feats//2,3)
        self.layer4 = BasicConv(n_feats, n_feats, 3, 1, 1)
        self.alise = BasicConv(2 * n_feats, n_feats, 1, 1, 0)  # 1*1卷积用于压缩通道数
        self.atten = CALayer(n_feats)  # 通道注意力机制
        self.weight1 = common.Scale(1)
        self.weight2 = common.Scale(1)
        self.weight3 = common.Scale(1)
        self.weight4 = common.Scale(1)
        self.weight5 = common.Scale(1)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        # pdb.set_trace()
        x4 = self.layer4(self.atten(self.alise(torch.cat([self.weight2(x2), self.weight3(x1)], 1))))  #
        return self.weight4(x) + self.weight5(x4)


class Updownblock(nn.Module):  # HPB模块
    def __init__(self, n_feats):
        super(Updownblock, self).__init__()
        self.encoder = one_module(n_feats)  # arfb模块
        self.decoder_low = CMTBlock(n_feats)
        self.decoder_high = one_module(n_feats)
        self.alise = one_module(n_feats)
        self.alise2 = BasicConv(2 * n_feats, n_feats, 1, 1, 0)  # one_module(n_feats)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.att = CALayer(n_feats)

    def forward(self, x):
        x1 = self.encoder(x)  # arfb
        x2 = self.down(x1)  # 全局平局池化
        high = x1 - F.interpolate(x2, size=x.size()[-2:], mode='bilinear', align_corners=True)  # 相减提取高频信息
        for i in range(5):
            x2 = self.decoder_low(x2)
        x3 = x2
        # x3 = self.decoder_low(x2)
        high1 = self.decoder_high(high)
        x4 = F.interpolate(x3, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return self.alise(self.att(self.alise2(torch.cat([x4, high1], dim=1)))) + x


class ResBlock(nn.Module):  # 整个arfb模块
    def __init__(self, n_feats):
        super(ResBlock, self).__init__()
        self.layer1 = one_conv(n_feats, n_feats // 2, 3)  # 图6中第一个ru
    def forward(self, x):
        x1 = self.layer1(x)
        return x1


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class Un(nn.Module):
    def __init__(self, n_feats, wn):
        super(Un, self).__init__()
        # self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=3 >> 1, stride=1)
        # self.resblock = ResBlock(n_feats)
        #
        # self.decoder_low1 = EfficientViTBlockyuanshi(in_channels=n_feats, dim=8,expand_ratio=4, norm="bn2d", act_func="hswish" )
        #
        self.att = CALayer(n_feats)
        # self.alise0 = common.default_conv(n_feats, n_feats, 3)
        # self.alise1 = common.default_conv(n_feats, n_feats, 3)
        # self.alise2 = common.default_conv(n_feats, n_feats, 3)
        self.alise3 = common.default_conv(n_feats, n_feats, 3)
        # self.reduce22 = common.default_conv(2 * n_feats, n_feats, 1)
        # self.reduce21 = common.default_conv(2 * n_feats, n_feats, 1)
        # self.reduce3 = common.default_conv(3 * n_feats, n_feats, 1)
        # self.reduce4 = common.default_conv(4 * n_feats, n_feats, 1)
        # self.reduce5 = common.default_conv(5 * n_feats, n_feats, 1)
        # self.reduce61 = common.default_conv(6 * n_feats, n_feats, 1)
        # self.reduce62 = common.default_conv(6 * n_feats, n_feats, 1)
        # self.fftblock = nn.Sequential(,
        #                               torch.fft.rfftn(dim=[2, 3]),
        #                               nn.Conv2d(n_feats, n_feats, 3, ),
        #                               nn.LeakyReLU(),
        #                               torch.fft.irfftn(dim=[2, 3]))
        # self.fftblockcon1layer1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        # self.fftblockcon2layer1 = ComplexConv2d(n_feats, n_feats, 3, padding=1)
        # self.fftblockactlayer1 = complex_relu()

        # self.fftblockcon1layer2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        # self.fftblockcon2layer2 = ComplexConv2d(n_feats, n_feats, 3, padding=1)
        # # self.fftblockactlayer2 = nn.LeakyReLU()
        #
        # self.fftblockcon1layer3 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        # self.fftblockcon2layer3 = ComplexConv2d(n_feats, n_feats, 3, padding=1)
        # # self.fftblockactlayer3 = nn.LeakyReLU()
        #
        # self.fftblockcon1layer4 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        # self.fftblockcon2layer4 = ComplexConv2d(n_feats, n_feats, 3, padding=1)
        # # self.fftblockactlayer4 = nn.LeakyReLU()
        #
        # self.fftblockcon1layer5 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        # self.fftblockcon2layer5 = ComplexConv2d(n_feats, n_feats, 3, padding=1)
        # # self.fftblockactlayer5 = nn.LeakyReLU()

        self.weightx = common.Scale(1)
        self.weighttrans = common.Scale(1)
        self.weightx10 = common.Scale(1)

        self.G0 = n_feats
        self.G = n_feats
        self.C = 4
        self.D = 7

        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        self.decoder_low = nn.ModuleList([EfficientViTBlockyuanshi(in_channels=n_feats, dim=8,expand_ratio=4, norm="bn2d", act_func="hswish" )])
        self.fftblockcon1 = nn.ModuleList([nn.Conv2d(n_feats, n_feats, 3, padding=1)])
        self.fftblockcon2 = nn.ModuleList([ComplexConv2d(n_feats, n_feats, 3, padding=1)])
        self.alise1 = nn.ModuleList([common.default_conv(n_feats, n_feats, 3)])
        self.resblock = nn.ModuleList([ResBlock(n_feats)])
        self.alise2 = nn.ModuleList([common.default_conv(n_feats, n_feats, 3)])
        self.reduce22 = nn.ModuleList([common.default_conv(2 * n_feats, n_feats, 1)])

        for _ in range(self.D - 1):
            self.alise1.append(common.default_conv(n_feats, n_feats, 3))

        for _ in range(self.D - 1):
            self.resblock.append(ResBlock(n_feats))

        for _ in range(self.D - 1):
            self.alise2.append(common.default_conv(n_feats, n_feats, 3))

        for _ in range(self.D - 1):
            self.reduce22.append(common.default_conv(2 * n_feats, n_feats, 1))

        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        for _ in range(self.D -1):
            self.decoder_low.append(EfficientViTBlockyuanshi(in_channels=n_feats, dim=8,expand_ratio=4, norm="bn2d", act_func="hswish" ))

        for _ in range(self.D -1):
            self.fftblockcon1.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))

        for _ in range(self.D -1):
            self.fftblockcon2.append(ComplexConv2d(n_feats, n_feats, 3, padding=1))


        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )



    def forward(self, x):

        b, c, h, w = x.size()

        xchushi = x

        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)

            xatt = self.decoder_low[i](x)

            xattfft1 = self.fftblockcon1[i](xatt)

            xattfft2 = torch.fft.rfftn(xattfft1, dim=[2, 3])
            xattfft2 = self.fftblockcon2[i](xattfft2)
            xattfft2 = complex_relu(xattfft2)
            xattfft2 = torch.fft.irfftn(xattfft2, dim=[2, 3])

            xattfft = xattfft1 + xattfft2
            xattfft = self.alise1[i](xattfft)

            xres = self.resblock[i](x)

            xatttotal = self.alise2[i](self.reduce22[i](torch.cat([xres, xattfft], dim=1)))

            x = x+xatttotal

            local_features.append(x)

        x10 = self.gff(torch.cat(local_features, 1))  # global residual learning


        # x11=x10
        #
        # x11q = self.fftblockcon1layer1(x11)
        # x11q = torch.fft.rfftn(x11q, dim=[2, 3])
        # x11q = self.fftblockcon2layer1(x11q)
        # x11q = complex_relu(x11q)
        # x11q = torch.fft.irfftn(x11q, dim=[2, 3])
        #
        # x11q = x11q + x11
        # x11q = self.alise1(x11q)
        # x11 = self.resblock(x11)
        # x11 = self.alise2(self.reduce22(torch.cat([x11, x11q], dim=1)))
        #
        # for i in range(5):
        #     x11 = self.alise0(x11)
        #     x11 = self.decoder_low1(x11)

        return self.weighttrans(self.alise3(self.att(x10))) + self.weightx(xchushi)


class myCmt(nn.Module):
    def __init__(self, upscale=4, conv=common.default_conv):
        super(myCmt, self).__init__()
        wn = lambda x: torch.nn.utils.weight_norm(x)
        n_feats = 64
        n_blocks = 1
        kernel_size = 3
        scale = upscale  # args.scale[0] #gaile
        # act = nn.ReLU(True)
        # self.up_sample = F.interpolate(scale_factor=2, mode='nearest')
        self.n_blocks = n_blocks

        # define head module
        modules_head = [conv(4, n_feats, kernel_size)]

        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(
                Un(n_feats=n_feats, wn=wn))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 4, kernel_size)
        ]

        self.up = nn.Sequential(common.Upsampler(conv, scale, n_feats, act=False),
                                BasicConv(n_feats, 4, 3, 1, 1))
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)
        self.reduce = conv(n_blocks * n_feats, n_feats, kernel_size)

    def forward(self, x1, x2=None, test=False):
        # x1 = self.sub_mean(x1)
        x1 = self.head(x1)  # 第一个卷积层
        res2 = x1
        # res2 = x2
        body_out = []
        for i in range(self.n_blocks):
            x1 = self.body[i](x1)
            body_out.append(x1)
        res1 = torch.cat(body_out, 1)
        res1 = self.reduce(res1)

        x1 = self.tail(res1)
        x1 = self.up(res2) + x1

        return x1

    # def load_state_dict(self, state_dict, strict=False):
    #     own_state = self.state_dict()
    #     for name, param in state_dict.items():
    #         if name in own_state:
    #             if isinstance(param, nn.Parameter):
    #                 param = param.data
    #             try:
    #                 own_state[name].copy_(param)
    #             except Exception:
    #                 if name.find('tail') >= 0:
    #                     print('Replace pre-trained upsampler to new one...')
    #                 else:
    #                     raise RuntimeError('While copying the parameter named {}, '
    #                                        'whose dimensions in the model are {} and '
    #                                        'whose dimensions in the checkpoint are {}.'
    #                                        .format(name, own_state[name].size(), param.size()))
    #         elif strict:
    #             if name.find('tail') == -1:
    #                 raise KeyError('unexpected key "{}" in state_dict'
    #                                .format(name))
    #
    #     if strict:
    #         missing = set(own_state.keys()) - set(state_dict.keys())
    #         if len(missing) > 0:
    #             raise KeyError('missing keys in state_dict: "{}"'.format(missing))
    #     # MSRB_out = []from model import common


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__=='__main__':
    device = torch.device('cuda')
    cmt = myCmt(upscale=2).to(device)
    summary(cmt,input_size=(4,128,128))

    print('parameters_count:', count_parameters(cmt))

    # -- coding: utf-8 --
    import torch
    import torchvision
    from thop import profile

    # Model
    print('==> Building model..')


    dummy_input = torch.randn(1, 4, 128,128).cuda()
    flops, params = profile(cmt, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
