import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F

import sys
#from models.model_variants import PATNetwork
from .model_variants import PATNetwork

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'batch_sync':
        norm_layer = BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal',
             gpu_ids=[], n_downsampling=2):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'PATN':
        assert len(input_nc) == 2
        # Changed n_blocks from 9 to 7
        netG = PATNetwork(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                           n_blocks=11, gpu_ids=gpu_ids, n_downsampling=n_downsampling)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[], use_dropout=False,
             n_downsampling=2):

    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netD == 'resnet':
        netD = ResnetDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_layers_D,
                                       gpu_ids=[], padding_type='reflect', use_sigmoid=use_sigmoid,
                                       n_downsampling=n_downsampling)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)

    if use_gpu:
        netD.cuda(gpu_ids[0])

    return netD

    # elif which_model_netD == 'patchgan':
    #     netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    #     init_gain = 0.02
    #     return init_net(netD, init_type, init_gain, gpu_ids)





def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################    

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetDiscriminator(nn.Module):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[],
                 padding_type='reflect', use_sigmoid=False, n_downsampling=2):
        assert (n_blocks >= 0)
        super(ResnetDiscriminator, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # n_downsampling = 2
        if n_downsampling <= 2:
            for i in range(n_downsampling):
                mult = 2 ** i
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                    stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
        elif n_downsampling == 3:
            mult = 2 ** 0
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            mult = 2 ** 1
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
            mult = 2 ** 2
            model += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult),
                      nn.ReLU(True)]

        if n_downsampling <= 2:
            mult = 2 ** n_downsampling
        else:
            mult = 4
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        if use_sigmoid:
            model += [nn.Sigmoid()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

    # class GANLoss(nn.Module):
    #     """Define different GAN objectives.
    #     The GANLoss class abstracts away the need to create the target label tensor
    #     that has the same size as the input.
    #     """
    #
    #     def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
    #         """ Initialize the GANLoss class.
    #         Parameters:
    #             gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
    #             target_real_label (bool) - - label for a real image
    #             target_fake_label (bool) - - label of a fake image
    #         Note: Do not use sigmoid as the last layer of Discriminator.
    #         LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
    #         """
    #         super(GANLoss, self).__init__()
    #         self.register_buffer('real_label', torch.tensor(target_real_label))
    #         self.register_buffer('fake_label', torch.tensor(target_fake_label))
    #         self.gan_mode = gan_mode
    #         if gan_mode == 'lsgan':
    #             self.loss = nn.MSELoss()
    #         elif gan_mode == 'vanilla':
    #             self.loss = nn.BCEWithLogitsLoss()
    #         elif gan_mode in ['wgangp']:
    #             self.loss = None
    #         else:
    #             raise NotImplementedError('gan mode %s not implemented' % gan_mode)
    #
    #     def get_target_tensor(self, prediction, target_is_real):
    #         """Create label tensors with the same size as the input.
    #         Parameters:
    #             prediction (tensor) - - tpyically the prediction from a discriminator
    #             target_is_real (bool) - - if the ground truth label is for real images or fake images
    #         Returns:
    #             A label tensor filled with ground truth label, and with the size of the input
    #         """
    #
    #         if target_is_real:
    #             target_tensor = self.real_label
    #         else:
    #             target_tensor = self.fake_label
    #         return target_tensor.expand_as(prediction)
    #
    #     def __call__(self, prediction, target_is_real):
    #         """Calculate loss given Discriminator's output and grount truth labels.
    #         Parameters:
    #             prediction (tensor) - - tpyically the prediction output from a discriminator
    #             target_is_real (bool) - - if the ground truth label is for real images or fake images
    #         Returns:
    #             the calculated loss.
    #         """
    #         if self.gan_mode in ['lsgan', 'vanilla']:
    #             target_tensor = self.get_target_tensor(prediction, target_is_real).cuda()
    #             loss = self.loss(prediction, target_tensor)
    #         elif self.gan_mode == 'wgangp':
    #             if target_is_real:
    #                 loss = -prediction.mean()
    #             else:
    #                 loss = prediction.mean()
    #         return loss

    # def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    #     """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    #     Parameters:
    #         net (network)      -- the network to be initialized
    #         init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
    #         gain (float)       -- scaling factor for normal, xavier and orthogonal.
    #         gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    #     Return an initialized network.
    #     """
    #     if len(gpu_ids) > 0:
    #         assert(torch.cuda.is_available())
    #         net.to(gpu_ids[0])
    #         net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    #     init_weightsD(net, init_type, init_gain=init_gain)
    #     return net

    # def init_weightsD(net, init_type='normal', init_gain=0.02):
    #     """Initialize network weights.
    #     Parameters:
    #         net (network)   -- network to be initialized
    #         init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
    #         init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    #     We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    #     work better for some applications. Feel free to try yourself.
    #     """
    #     def init_func(m):  # define the initialization function
    #         classname = m.__class__.__name__
    #         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
    #             if init_type == 'normal':
    #                 init.normal_(m.weight.data, 0.0, init_gain)
    #             elif init_type == 'xavier':
    #                 init.xavier_normal_(m.weight.data, gain=init_gain)
    #             elif init_type == 'kaiming':
    #                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    #             elif init_type == 'orthogonal':
    #                 init.orthogonal_(m.weight.data, gain=init_gain)
    #             else:
    #                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    #             if hasattr(m, 'bias') and m.bias is not None:
    #                 init.constant_(m.bias.data, 0.0)
    #         elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
    #             init.normal_(m.weight.data, 1.0, init_gain)
    #             init.constant_(m.bias.data, 0.0)
    #
    #     print('initialize network with %s' % init_type)
    #     net.apply(init_func)  # apply the initialization function <init_func>