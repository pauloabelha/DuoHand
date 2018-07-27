import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def _print_layer_output_shape(layer_name, output_shape):
    print("Layer " + layer_name + " output shape: " + str(output_shape))

def VargoNetConvBlock(kernel_size, stride, filters, in_channels, padding=0):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(num_features=filters)
        )

def VargoNetResConvSequence(stride, filters1, filters2,
                          padding1=1, padding2=0, padding3=0,
                          first_in_channels=0):
    if first_in_channels == 0:
        first_in_channels = filters1
    return nn.Sequential(
        # added padding = 1 to make shapes fit when joining
        # with left module
        VargoNetConvBlock(kernel_size=1, stride=stride, filters=filters1,
                        in_channels=first_in_channels, padding=padding1),
        nn.ReLU(),
        VargoNetConvBlock(kernel_size=3, stride=1, filters=filters1,
                        in_channels=filters1, padding=padding2),
        nn.ReLU(),
        VargoNetConvBlock(kernel_size=1, stride=1, filters=filters2,
                        in_channels=filters1, padding=padding3)
    )

class VargoNetResBlockIDSkip(nn.Module):
    def __init__(self, filters1, filters2,
                 padding_right1=1, padding_right2=0, padding_right3=0):
        super(VargoNetResBlockIDSkip, self).__init__()
        self.right_res = VargoNetResConvSequence(stride=1,
                                               filters1=filters1,
                                               filters2=filters2,
                                               padding1=padding_right1,
                                               padding2=padding_right2,
                                               padding3=padding_right3,
                                               first_in_channels=
                                               filters2)
        self.relu = nn.ReLU()

    def forward(self, input):
        left_res = input
        right_res = self.right_res(input)
        # element-wise sum
        out = left_res + right_res
        out = self.relu(out)
        return out

class VargoNetResBlockConv(nn.Module):
    def __init__(self, stride, filters1, filters2, first_in_channels=0,
                 padding_left=0, padding_right1=0, padding_right2=0,
                 padding_right3=0):
        super(VargoNetResBlockConv, self).__init__()
        if first_in_channels == 0:
            first_in_channels = filters1
        self.left_res = VargoNetConvBlock(kernel_size=1, stride=stride,
                                        filters=filters2,
                                        padding=padding_left,
                                        in_channels=first_in_channels)
        self.right_res = VargoNetResConvSequence(stride=stride,
                                               filters1=filters1,
                                               filters2=filters2,
                                               padding1=padding_right1,
                                               padding2=padding_right2,

                                               padding3=padding_right3,
                                               first_in_channels=
                                               first_in_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        left_res = self.left_res(input)
        right_res = self.right_res(input)
        # element-wise sum
        out = left_res + right_res
        out = self.relu(out)
        return out

def make_bilinear_weights(size, num_channels):
    ''' Make a 2D bilinear kernel suitable for upsampling
    Stack the bilinear kernel for application to tensor '''
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, 1, size, size)
    for i in range(num_channels):
        w[i, 0] = filt
    return w



class SoftmaxLogProbability2D(torch.nn.Module):
    def __init__(self):
        super(SoftmaxLogProbability2D, self).__init__()

    def forward(self, x):
        orig_shape = x.data.shape
        seq_x = []
        for channel_ix in range(orig_shape[1]):
            softmax_ = F.softmax(x[:, channel_ix, :, :].contiguous()
                                 .view((orig_shape[0], orig_shape[2] * orig_shape[3])), dim=1)\
                .view((orig_shape[0], orig_shape[2], orig_shape[3]))
            seq_x.append(softmax_.log())
        x = torch.stack(seq_x, dim=1)
        return x

def parse_model_param(params_dict, key, default_value):
    try:
        ret = params_dict[key]
    except:
        if default_value == "Mandatory":
            # raise error again by trying to access value
            ret = params_dict[key]
        ret = default_value
    return ret

def cudafy(object, use_cuda):
    if use_cuda:
        return object.cuda()
    else:
        return object

class VargoNet(nn.Module):
    obj_channel = False
    use_rgbd = True
    # if RGB, 3; if RGB-D, 4
    in_channels = 4
    cross_entropy = True
    heatmap_ixs = range(3)
    num_heatmaps = 16
    use_cuda = None
    WEIGHT_LOSS_INTERMED1 = 0.5
    WEIGHT_LOSS_INTERMED2 = 0.5
    WEIGHT_LOSS_INTERMED3 = 0.5
    WEIGHT_LOSS_MAIN = 1



    def __init__(self, params_dict):
        self.use_rgbd = params_dict['use_rgbd']
        if self.use_rgbd:
            self.in_channels = 4
        else:
            self.in_channels = 3
        if params_dict['obj_channel']:
            self.obj_channel = True
            self.in_channels += 10
        super(VargoNet, self).__init__()
        # initialize variables
        self.use_cuda = parse_model_param(params_dict, 'use_cuda', default_value=False)
        # build network
        self.conv1 = cudafy(VargoNetConvBlock(kernel_size=7, stride=1, filters=64,
                                     in_channels=self.in_channels, padding=3), self.use_cuda)
        self.mp1 = cudafy(nn.MaxPool2d(kernel_size=3, stride=2, padding=1), self.use_cuda)
        self.res2a = cudafy(VargoNetResBlockConv(stride=1, filters1=64, filters2=256,
                                        padding_right1=1), self.use_cuda)
        self.res2b = cudafy(VargoNetResBlockIDSkip(filters1=64, filters2=256), self.use_cuda)
        self.res2c = cudafy(VargoNetResBlockIDSkip(filters1=64, filters2=256), self.use_cuda)
        self.res3a = cudafy(VargoNetResBlockConv(stride=2, filters1=128, filters2=512,
                                               padding_right3=1, first_in_channels=256), self.use_cuda)
        self.interm_loss1 = cudafy(VargoNetConvBlock(kernel_size=3, stride=1, filters=self.num_heatmaps,
                                                   in_channels=512, padding=1), self.use_cuda)
        self.interm_loss1_deconv = cudafy(nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True), self.use_cuda)
        self.interm_loss1_softmax = cudafy(SoftmaxLogProbability2D(), self.use_cuda)
        self.res3b = cudafy(VargoNetResBlockIDSkip(filters1=128, filters2=512), self.use_cuda)
        self.res3c = cudafy(VargoNetResBlockIDSkip(filters1=128, filters2=512), self.use_cuda)
        self.res4a = cudafy(VargoNetResBlockConv(stride=2, filters1=256, filters2=1024,
                                        padding_right3=1,
                                        first_in_channels=512), self.use_cuda)
        self.interm_loss2 = cudafy(VargoNetConvBlock(kernel_size=3, stride=1,
                                            filters=self.num_heatmaps, in_channels=1024,
                                            padding=1), self.use_cuda)
        self.interm_loss2_deconv = cudafy(nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True), self.use_cuda)
        self.interm_loss2_softmax = cudafy(SoftmaxLogProbability2D(), self.use_cuda)
        self.res4b = cudafy(VargoNetResBlockIDSkip(filters1=256, filters2=1024), self.use_cuda)
        self.res4c = cudafy(VargoNetResBlockIDSkip(filters1=256, filters2=1024), self.use_cuda)
        self.res4d = cudafy(VargoNetResBlockIDSkip(filters1=256, filters2=1024), self.use_cuda)
        self.conv4e = cudafy(VargoNetConvBlock(kernel_size=3, stride=1, filters=512,
                                     in_channels=1024, padding=1), self.use_cuda)
        self.interm_loss3 = cudafy(VargoNetConvBlock(kernel_size=3, stride=1,
                                            filters=self.num_heatmaps, in_channels=512,
                                            padding=1), self.use_cuda)
        self.interm_loss3_deconv = cudafy(nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True), self.use_cuda)
        self.interm_loss3_softmax = cudafy(SoftmaxLogProbability2D(), self.use_cuda)
        self.conv4f = cudafy(VargoNetConvBlock(kernel_size=3, stride=1, filters=256,
                                      in_channels=512, padding=1), self.use_cuda)
        self.main_loss_conv = cudafy(VargoNetConvBlock(kernel_size=3, stride=1,
                                              filters=self.num_heatmaps, in_channels=256,
                                              padding=1), self.use_cuda)
        self.main_loss_deconv = cudafy(nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True), self.use_cuda)
        self.softmax_final = cudafy(SoftmaxLogProbability2D(), self.use_cuda)

    def forward_common_net(self, x):
        out = self.conv1(x)
        out = self.mp1(out)
        out = self.res2a(out)
        out = self.res2b(out)
        out = self.res2c(out)
        res3aout = self.res3a(out)
        out = self.res3b(res3aout)
        out = self.res3c(out)
        res4aout = self.res4a(out)
        out = self.res4b(res4aout)
        out = self.res4c(out)
        out = self.res4d(out)
        conv4eout = self.conv4e(out)
        conv4fout = self.conv4f(conv4eout)
        return res3aout, res4aout, conv4eout, conv4fout

    def forward_subnet(self, x):
        res3aout, res4aout, conv4eout, conv4fout = self.forward_common_net(x)
        # intermediate losses
        # intermed 1
        out_intermed1 = self.interm_loss1(res3aout)
        out_intermed1 = self.interm_loss1_deconv(out_intermed1)
        if self.cross_entropy:
            out_intermed1 = self.interm_loss1_softmax(out_intermed1)
        # intermed 2
        out_intermed2 = self.interm_loss2(res4aout)
        out_intermed2 = self.interm_loss2_deconv(out_intermed2)
        if self.cross_entropy:
            out_intermed2 = self.interm_loss2_softmax(out_intermed2)
        # intermed 3
        out_intermed3 = self.interm_loss3(conv4eout)
        out_intermed3 = self.interm_loss3_deconv(out_intermed3)
        if self.cross_entropy:
            out_intermed3 = self.interm_loss3_softmax(out_intermed3)
        return out_intermed1, out_intermed2, out_intermed3, conv4fout,\
               res3aout, res4aout, conv4eout

    def forward_main_loss(self, conv4fout):
        out = self.main_loss_conv(conv4fout)
        out = self.main_loss_deconv(out)
        out_main = out
        # main loss
        if self.cross_entropy:
            out_main = self.softmax_final(out)
        return out_main

    def forward(self, x):
        # get subVargoNet outputs (common to JORNet)
        out_intermed1, out_intermed2, out_intermed3, conv4fout, _, _, _ = self.forward_subnet(x)
        # out to main loss of VargoNet
        out_main = self.forward_main_loss(conv4fout)
        return out_intermed1, out_intermed2, out_intermed3, out_main