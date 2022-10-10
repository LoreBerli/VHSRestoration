# the unet code is inspired from https://github.com/usuyama/pytorch-unet

import math
from builtins import super
import torch
import torch.nn as nn
from torch.nn import functional as F
from modules import ASPP as ASPP
from modules import ASPP_DWISE as ASPP_DWISE
from modules import Squeeze_Excite_Block as Squeeze_Excite_Block
from modules import AttentionBlock as AttentionBlock


class SimpleResNet(nn.Module):
    def __init__(self, n_filters, n_blocks,upscale=3,downsample=1.0):
        super(SimpleResNet, self).__init__()
        self.conv1 = UnetBlock(in_channels=3, out_channels=n_filters, use_residual=True, use_bn=False)
        convblock = [UnetBlock(in_channels=n_filters, out_channels=n_filters, use_residual=True, use_bn=False) for _ in
                     range(n_blocks - 1)]
        self.convblocks = nn.Sequential(*convblock)
        self.sr = sr_espcn(n_filters, scale_factor=upscale, out_channels=3)
        self.upscale = nn.Upsample(scale_factor=upscale, mode='bicubic', align_corners=True)
        self.downsample = nn.Upsample(scale_factor=downsample, mode='bicubic', align_corners=True)
        self.clip = nn.Hardtanh()

    def forward(self, input):
        x = self.conv1(input)
        x = self.convblocks(x)
        x = self.sr(x)
        x = self.clip(x + self.upscale(input))
        return self.downsample(x)

    # def reparametrize(self):
    #     for block in self.convblocks:
    #         if hasattr(block, 'conv_adapter'):
    #             block.reparametrize_convs()

class RDB(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(RDB, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class UnetBlock_plus(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, stride=1, kernel_size=3, use_residual=True,second_kernel=7):
        super(UnetBlock_plus, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_reparametrized = False
        self.use_residual = use_residual

        # if in_channels == out_channels and use_residual:
        self.conv_adapter = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        self.conv_final = nn.Conv2d(in_channels * 3, out_channels, kernel_size, padding=kernel_size // 2, stride=stride)
        self.squeezer = Squeeze_Excite_Block(in_channels*2,reduction=8)
        self.conv_1_1_expander = nn.Conv2d(in_channels, in_channels * 2, 1, padding=0, stride=1,bias=False)
        self.conv_depthwise_separable = nn.Conv2d(in_channels * 2, in_channels * 2, second_kernel, padding=second_kernel // 2, stride=stride, groups=in_channels, padding_mode='zeros',bias=False)
        self.conv_1_1_contractor = nn.Conv2d(in_channels * 2, in_channels, 1, padding=0, stride=1,bias=False)

        self.conv20 = nn.Conv2d(in_channels*2, in_channels, 3, padding=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn0 = nn.BatchNorm2d(in_channels*2)
        # self.act = nn.ReLU6()3
        self.act = nn.LeakyReLU(0.2, inplace=True) # used by srgan_128x128_94_10-01-2021_0917_valloss0.30919_mobilenet_flickr.pkl
        #self.act = nn.ReLU(inplace=True)

    def forward(self, input):

        expanded_1_1 =  self.conv_1_1_expander(input)
        expanded_1_1 = self.bn0(expanded_1_1)
        depth_wise_2C = self.conv_depthwise_separable(expanded_1_1)
        depth_wise_2C = F.leaky_relu(self.squeezer(depth_wise_2C), 0.2)
        contracted_1_1 = self.conv_1_1_contractor(depth_wise_2C)
        contracted_1_1 = torch.cat([input, contracted_1_1], 1)
        pre2 = F.leaky_relu(self.conv20(contracted_1_1),0.2)
        pre2 = torch.cat([contracted_1_1,pre2], 1)
        #depth_wise_2C = F.layer_norm(depth_wise_2C,depth_wise_2C.shape[1:])#self.bn0(depth_wise_2C)


        x = self.conv_final(pre2) # se is_reparametrized==true allora questa diventa l'equazione 2
        x = F.leaky_relu(x,0.2)
        # if self.use_residual and not self.is_reparametrized:
        #     if self.in_channels == self.out_channels:
        #         out =x*0.5+ input + self.conv_adapter(input)*0.5 # equazione 2
        #     else:
        #         out=x
        #x = F.layer_norm(x,x.shape[1:])#self.bn(x)
        if self.in_channels == self.out_channels:
            x = x+input
        return x

    def reparametrize_convs(self):
        identity_conv = nn.init.dirac_(torch.empty_like(self.conv_final.weight))
        padded_adapter_conv = F.pad(self.conv_adapter.weight, (1, 1, 1, 1),"constant",0) #padding della conv 1x1 (diventera un kernel 3x3)
        #
        if self.in_channels == self.out_channels:
            new_conv_weights = self.conv_final.weight + padded_adapter_conv + identity_conv # equazione 3
            new_conv_bias = self.conv_final.bias + self.conv_adapter.bias

            self.conv_final.weight.data = new_conv_weights
            self.conv_final.bias.data = new_conv_bias

        self.is_reparametrized = True

class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, stride=1, kernel_size=3, use_residual=True):
        super(UnetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_reparametrized = False
        self.use_residual = use_residual

        # if in_channels == out_channels and use_residual:
        self.conv_adapter = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        # self.act = nn.ReLU6()
        # self.act = nn.LeakyReLU(0.2, inplace=True) # used by srgan_128x128_94_10-01-2021_0917_valloss0.30919_mobilenet_flickr.pkl
        self.act = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input) # se is_reparametrized==true allora questa diventa l'equazione 2
        if self.use_residual and not self.is_reparametrized:
            if self.in_channels == self.out_channels:
                x += input + self.conv_adapter(input) # equazione 2
        x = self.bn(x)
        x = self.act(x)
        return x

    def reparametrize_convs(self):
        identity_conv = nn.init.dirac_(torch.empty_like(self.conv1.weight))
        padded_adapter_conv = F.pad(self.conv_adapter.weight, (1, 1, 1, 1), "constant", 0) #padding della conv 1x1 (diventera un kernel 3x3)
        #
        if self.in_channels == self.out_channels:
            new_conv_weights = self.conv1.weight + padded_adapter_conv + identity_conv # equazione 3
            new_conv_bias = self.conv1.bias + self.conv_adapter.bias

            self.conv1.weight.data = new_conv_weights
            self.conv1.bias.data = new_conv_bias

        self.is_reparametrized = True

def layer_generator_plus(in_channels, out_channels, use_batch_norm=False, use_bias=True, residual_block=True, n_blocks=3, inverted=False):
    n_blocks = int(n_blocks)
    second_kernels=[7,5,3,3] if inverted else [7,3,3,3] #7,5,3  3,5,7

    first_layer = UnetBlock_plus(in_channels=in_channels, out_channels=out_channels, use_bn=use_batch_norm,
                                 use_residual=residual_block, second_kernel=second_kernels[0])

    return nn.Sequential(*(
            [first_layer] + [
        UnetBlock_plus(in_channels=out_channels, out_channels=out_channels, use_bn=use_batch_norm,
                       use_residual=residual_block, second_kernel=second_kernels[i+1])
        for i in range(n_blocks - 1)]
        # nn.Conv2d(out_channels, out_channels, 3, padding=1),
        # nn.ReLU(inplace=True)
    ))

def layer_generator(in_channels, out_channels, use_batch_norm=False, use_bias=True, residual_block=True, n_blocks=2,inverted=False):
    n_blocks = int(n_blocks)
    first_layer = UnetBlock(in_channels=in_channels, out_channels=out_channels, use_bn=use_batch_norm,
                            use_residual=residual_block)
    return nn.Sequential(*(
            [first_layer] + [
        UnetBlock(in_channels=out_channels, out_channels=out_channels, use_bn=use_batch_norm,
                  use_residual=residual_block)
        for _ in range(n_blocks - 1)]
        # nn.Conv2d(out_channels, out_channels, 3, padding=1),
        # nn.ReLU(inplace=True)
    ))

def sr_espcn(n_filters, scale_factor=2, out_channels=3, kernel_size=1):
    return nn.Sequential(*[
        nn.Conv2d(kernel_size=kernel_size, in_channels=n_filters, out_channels=(scale_factor ** 2) * out_channels,
                  padding=kernel_size // 2),
        nn.PixelShuffle(scale_factor),
    ])




class SARUnet(nn.Module):

    def __init__(self, in_dim=3, n_class=3, downsample=None, residual=False, batchnorm=False, scale_factor=2,
                 n_filters=64, layer_multiplier=1,is_ASPP_DWISE=True):
        """
        Args:
            in_dim (float, optional):
                channel dimension of the input
            n_class (str):
                channel dimension of the output
            n_filters (int, optional):
                maximum number of filters. the layers start with n_filters / 2,  after each layer this number gets multiplied by 2
                 during the encoding stage and until it reaches n_filters. During the decoding stage the number follows the reverse
                 scheme. Default is 64
            downsample (None or float, optional)
                can be used for downscaling the output. e.g., if you use downsample=0.5 the output resolution will be halved
            residual (bool):
                if using the residual scheme and adding the input to the final output
            scale_factor (int):
                upscale factor. if you want a rational upscale (e.g. 720p to 1080p, which is 1.5) combine it
                 with the downsample parameter
            layer_multiplier (int or float):
                compress or extend the network depth in terms of total layers. configured as a multiplier to the number of the
                basic blocks which composes the layers
            batchnorm (bool, default=False):
                whether use batchnorm or not. If True should decrease quality and performances.
        """

        super().__init__()

        self.residual = residual
        self.n_class = n_class
        self.scale_factor = scale_factor

        self.dconv_down1 = layer_generator_plus(in_dim, n_filters // 2, use_batch_norm=False,
                                                n_blocks=2 * layer_multiplier, inverted=True)
        self.dconv_down2 = layer_generator_plus(n_filters // 2, n_filters, use_batch_norm=batchnorm,
                                                n_blocks=3 * layer_multiplier, inverted=True)
        self.dconv_down3 = layer_generator_plus(n_filters, n_filters, use_batch_norm=batchnorm,
                                                n_blocks=3 * layer_multiplier, inverted=True)
        self.dconv_down4 = layer_generator_plus(n_filters, n_filters, use_batch_norm=batchnorm,
                                                n_blocks=4 * layer_multiplier, inverted=True)

        self.maxpool = nn.MaxPool2d(2)
        if downsample is not None and downsample != 1.0:
            #TODO scale factor hard-coded
            self.downsample = nn.Upsample(scale_factor=downsample, mode='bicubic', align_corners=True)
        else:
            self.downsample = nn.Identity()
        #TODO bilinear vs bicubic
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.aspp_bridge = ASPP_DWISE(n_filters, n_filters) if is_ASPP_DWISE else ASPP(n_filters,n_filters)

        self.squeeze_excite1 = Squeeze_Excite_Block(n_filters // 2)  # aggiunto squeeze_exicite

        self.squeeze_excite2 = Squeeze_Excite_Block(n_filters)  # aggiunto squeeze_exicite

        self.squeeze_excite3 = Squeeze_Excite_Block(n_filters)  # aggiunto squeeze_exicite

        self.attn3 = AttentionBlock(n_filters, n_filters, n_filters)

        self.attn2 = AttentionBlock(n_filters, n_filters, n_filters)

        self.attn1 = AttentionBlock(n_filters // 2, n_filters, n_filters)

        self.dconv_up3 = layer_generator_plus(n_filters + n_filters, n_filters, use_batch_norm=batchnorm,
                                              n_blocks=4 * layer_multiplier)
        self.dconv_up2 = layer_generator_plus(n_filters + n_filters, n_filters, use_batch_norm=batchnorm,
                                              n_blocks=3 * layer_multiplier)
        self.dconv_up1 = layer_generator_plus(n_filters + n_filters // 2, n_filters * 2, use_batch_norm=False,
                                              n_blocks=2 * layer_multiplier)

        self.layers = [self.dconv_down1, self.dconv_down2, self.dconv_down3, self.dconv_down4, self.dconv_up3,
                       self.dconv_up2, self.dconv_up1]

        sf = self.scale_factor

        self.to_rgb = nn.Conv2d(n_filters // 2, 3, kernel_size=1)
        if sf > 1:
            self.upsamp = nn.Upsample(scale_factor=sf)
            #self.conv_last = nn.Conv2d(3+n_filters,3+n_filters, kernel_size=1, padding=0,padding_mode='replicate')
            self.conv_last = nn.Conv2d(3+n_filters*2, n_class, kernel_size=3, padding=1)
            self.pixel_shuffle = nn.PixelShuffle(1)#nn.PixelShuffle(sf)
        else:
            self.conv_last = nn.Conv2d(n_filters // 2, 3, kernel_size=1)

    def forward(self, input):
        x = input

        conv1 = self.dconv_down1(x)
        conv1 = self.squeeze_excite1(conv1)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        conv2 = self.squeeze_excite2(conv2)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        conv3 = self.squeeze_excite3(conv3)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.aspp_bridge(x)

        x = self.attn3(conv3, x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.attn2(conv2, x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.attn1(conv1, x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)



        sf = self.scale_factor

        if sf > 1:
            x=self.upsamp(x)
            x = torch.cat([x,F.interpolate(input, scale_factor=sf,mode='bicubic')],1)
            x = self.conv_last(x)
            #x = self.pixel_shuffle(x)

        # x = self.to_rgb(x)
        # if self.residual:
        #     sf = self.scale_factor  # (self.scale_factor // (2 if self.use_s2d and self.scale_factor > 1 else 1))
        #     x += F.interpolate(input[:, -self.n_class:, :, :],
        #                        scale_factor=sf,
        #                        mode='bicubic')*0.5
        #     x = torch.clamp(x, min=-1, max=1)

        return torch.clamp(self.downsample(x), min=-1, max=1)  # self.downsample(x)

    def reparametrize(self):
        for layer in self.layers:
            for block in layer:
                if hasattr(block, 'conv_adapter'):
                    block.reparametrize_convs()



class SRUnet(nn.Module):

    def __init__(self, in_dim=3, n_class=3, downsample=0.889, residual=False, batchnorm=False, scale_factor=3,
                 n_filters=64, layer_multiplier=1):
        """
        Args:
            in_dim (float, optional):
                channel dimension of the input
            n_class (str):
                channel dimension of the output
            n_filters (int, optional):
                maximum number of filters. the layers start with n_filters / 2,  after each layer this number gets multiplied by 2
                 during the encoding stage and until it reaches n_filters. During the decoding stage the number follows the reverse
                 scheme. Default is 64
            downsample (None or float, optional)
                can be used for downscaling the output. e.g., if you use downsample=0.5 the output resolution will be halved
            residual (bool):
                if using the residual scheme and adding the input to the final output
            scale_factor (int):
                upscale factor. if you want a rational upscale (e.g. 720p to 1080p, which is 1.5) combine it
                 with the downsample parameter
            layer_multiplier (int or float):
                compress or extend the network depth in terms of total layers. configured as a multiplier to the number of the
                basic blocks which composes the layers
            batchnorm (bool, default=False):
                whether use batchnorm or not. If True should decrease quality and performances.
        """

        super().__init__()

        self.residual = residual
        self.n_class = n_class
        self.scale_factor = scale_factor

        self.dconv_down1 = layer_generator(in_dim, n_filters // 2, use_batch_norm=False,
                                           n_blocks=2 * layer_multiplier)
        self.dconv_down2 = layer_generator(n_filters // 2, n_filters, use_batch_norm=batchnorm,
                                           n_blocks=3 * layer_multiplier)
        self.dconv_down3 = layer_generator(n_filters, n_filters, use_batch_norm=batchnorm,
                                           n_blocks=3 * layer_multiplier)
        self.dconv_down4 = layer_generator(n_filters, n_filters, use_batch_norm=batchnorm,
                                           n_blocks=3 * layer_multiplier)

        self.maxpool = nn.MaxPool2d(2)
        if downsample is not None and downsample != 1.0:
            self.downsample = nn.Upsample(scale_factor=downsample, mode='bicubic', align_corners=True)
        else:
            self.downsample = nn.Identity()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.dconv_up3 = layer_generator(n_filters + n_filters, n_filters, use_batch_norm=batchnorm,
                                         n_blocks=3 * layer_multiplier)
        self.dconv_up2 = layer_generator(n_filters + n_filters, n_filters, use_batch_norm=batchnorm,
                                         n_blocks=3 * layer_multiplier)
        self.dconv_up1 = layer_generator(n_filters + n_filters // 2, n_filters // 2, use_batch_norm=False,
                                         n_blocks=3 * layer_multiplier)

        self.layers = [self.dconv_down1, self.dconv_down2, self.dconv_down3, self.dconv_down4, self.dconv_up3,
                       self.dconv_up2, self.dconv_up1]

        sf = int(self.scale_factor)

        self.to_rgb = nn.Conv2d(n_filters // 2, 3, kernel_size=1)
        if sf > 1:
            self.conv_last = nn.Conv2d(n_filters // 2, (sf ** 2) * n_class, kernel_size=1, padding=0)
            self.pixel_shuffle = nn.PixelShuffle(sf)
        else:
            self.conv_last = nn.Conv2d(n_filters // 2, 3, kernel_size=1)

    def forward(self, input):
        x = input

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)

        sf = int(self.scale_factor)

        if sf > 1:
            x = self.pixel_shuffle(x)

        # x = self.to_rgb(x)
        if self.residual:
            sf = self.scale_factor  # (self.scale_factor // (2 if self.use_s2d and self.scale_factor > 1 else 1))
            x += F.interpolate(input[:, -self.n_class:, :, :],
                               scale_factor=sf,
                               mode='bicubic')
            x = torch.clamp(x, min=-1, max=1)

        return torch.clamp(self.downsample(x), min=-1, max=1)  # self.downsample(x)

    def reparametrize(self):
        for layer in self.layers:
            for block in layer:
                if hasattr(block, 'conv_adapter'):
                    block.reparametrize_convs()



class UNet(nn.Module):

    def __init__(self, in_dim=3, n_class=3, n_filters=32, downsample=None, residual=True, batchnorm=False,
                 scale_factor=2):
        """
        Args
            in_dim (float, optional):
                channel dimension of the input
            n_class (str).
                channel dimension of the output
            n_filters (int, optional)
                number of filters of the first channel. after layer it gets multiplied by 2 during the encoding stage,
                and divided during the decoding
            downsample (None or float, optional):
                can be used for downscaling the output. e.g., if you use downsample=0.5 the output resolution will be halved
            residual (bool):
                if using the residual scheme and adding the input to the final output
            scale_factor (int):
                basic upscale factor. if you want a rational upscale (e.g. 720p to 1080p, which is 1.5) combine it
                with the downsample parameter
        """

        super().__init__()

        self.residual = residual
        self.n_class = n_class
        self.scale_factor = scale_factor

        self.dconv_down1 = layer_generator(in_dim, n_filters, use_batch_norm=False)
        self.dconv_down2 = layer_generator(n_filters, n_filters * 2, use_batch_norm=batchnorm, n_blocks=2)
        self.dconv_down3 = layer_generator(n_filters * 2, n_filters * 4, use_batch_norm=batchnorm, n_blocks=2)
        self.dconv_down4 = layer_generator(n_filters * 4, n_filters * 8, use_batch_norm=batchnorm, n_blocks=2)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.dconv_up3 = layer_generator(n_filters * 8 + n_filters * 4, n_filters * 4, use_batch_norm=batchnorm,
                                              n_blocks=2)
        self.dconv_up2 = layer_generator(n_filters * 4 + n_filters * 2, n_filters * 2, use_batch_norm=batchnorm,
                                              n_blocks=2)
        self.dconv_up1 = layer_generator(n_filters * 2 + n_filters, n_filters, use_batch_norm=False, n_blocks=2)

        sf = self.scale_factor#(self.scale_factor * (2 if self.use_s2d else 1))

        self.to_rgb = nn.Conv2d(n_filters, 3, kernel_size=1)
        if sf > 1:
            self.conv_last = nn.Sequential(nn.Conv2d(n_filters,n_filters,7,1,3,groups=n_filters),
                                            nn.Conv2d(n_filters, (sf ** 2) * n_class, kernel_size=1, padding=0))
            self.pixel_shuffle = nn.PixelShuffle(sf)
        else:
            self.conv_last = nn.Conv2d(n_filters, 3, kernel_size=1)

        if downsample is not None and downsample != 1.0:
            self.downsample = nn.Upsample(scale_factor=downsample, mode='bicubic', align_corners=True)
        else:
            self.downsample = nn.Identity()
        self.layers = [self.dconv_down1, self.dconv_down2, self.dconv_down3, self.dconv_down4, self.dconv_up3,
                       self.dconv_up2, self.dconv_up1]

    def forward(self, input):
        x = input

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)

        sf = self.scale_factor #(self.scale_factor * (2 if self.use_s2d else 1))

        if sf > 1:
            x = self.pixel_shuffle(x)
        if self.residual:
            sf = self.scale_factor  # (self.scale_factor // (2 if self.use_s2d and self.scale_factor > 1 else 1))
            x += F.interpolate(input[:, -self.n_class:, :, :],
                               scale_factor=sf,
                               mode='bicubic')
            x = torch.clamp(x, min=-1, max=1)

        return torch.clamp(self.downsample(x), min=-1, max=1)

    def reparametrize(self):
        for layer in self.layers:
            for block in layer:
                if hasattr(block, 'conv_adapter'):
                    block.reparametrize_convs()
