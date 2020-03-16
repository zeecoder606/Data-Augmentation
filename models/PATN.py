import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
# losses
from losses.L1_plus_perceptualLoss import L1_plus_perceptualLoss

import sys
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

class TransferModel(BaseModel):
    def name(self):
        return 'TransferModel'
    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_P1_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_BP1_set = self.Tensor(nb, opt.BP_input_nc, size, size)
        self.input_P2_set = self.Tensor(nb, opt.P_input_nc, size, size)
        self.input_BP2_set = self.Tensor(nb, opt.BP_input_nc, size, size)

        self.mask_1 = None
        self.mask_2 = None
        self.fg_img = None
        self.bg_img = None
        self.bg_fill = None
        self.out_patn_img = None
        self.target_img_bg = None
        self.target_img_fg = None
        self.source_fg_mask = None

        input_nc = [opt.P_input_nc, opt.BP_input_nc+opt.BP_input_nc]
        self.netG = networks.define_G(input_nc, opt.P_input_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                        n_downsampling=opt.G_n_downsampling)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if opt.with_D_PB:
                self.netD_PB = networks.define_D(opt.P_input_nc+opt.BP_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

            if opt.with_D_PP:
                self.netD_PP = networks.define_D(opt.P_input_nc+opt.P_input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                            not opt.no_dropout_D,
                                            n_downsampling = opt.D_n_downsampling)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'netG', which_epoch)
            if self.isTrain:
                if opt.with_D_PB:
                    self.load_network(self.netD_PB, 'netD_PB', which_epoch)
                if opt.with_D_PP:
                    self.load_network(self.netD_PP, 'netD_PP', which_epoch)


        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_PP_pool = ImagePool(opt.pool_size)
            self.fake_PB_pool = ImagePool(opt.pool_size)
            # define loss functions
            # if opt.which_model_netD == 'patchgan':
            #     self.criterionGAN = networks.GANLoss(gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0)
            # else:
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.loss_l1 = torch.nn.L1Loss()
            if opt.L1_type == 'origin':
                self.criterionL1 = torch.nn.L1Loss()
            elif opt.L1_type == 'l1_plus_perL1':
                self.criterionL1 = L1_plus_perceptualLoss(opt.lambda_A, opt.lambda_B, opt.perceptual_layers, self.gpu_ids, opt.percep_is_l1)
            else:
                raise Exception('Unsurportted type of L1!')
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PB:
                self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            if opt.with_D_PP:
                self.optimizer_D_PP = torch.optim.Adam(self.netD_PP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            if opt.with_D_PB:
                self.optimizers.append(self.optimizer_D_PB)
            if opt.with_D_PP:
                self.optimizers.append(self.optimizer_D_PP)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            if opt.with_D_PB:
                networks.print_network(self.netD_PB)
            if opt.with_D_PP:
                networks.print_network(self.netD_PP)
        print('-----------------------------------------------')

    def set_input(self, input, opt):
        input_P1, input_BP1 = input['P1'], input['BP1']
        input_P2, input_BP2 = input['P2'], input['BP2']
        self.input_P1_set.resize_(input_P1.size()).copy_(input_P1)
        self.input_BP1_set.resize_(input_BP1.size()).copy_(input_BP1)
        self.input_P2_set.resize_(input_P2.size()).copy_(input_P2)
        self.input_BP2_set.resize_(input_BP2.size()).copy_(input_BP2)
        if opt.phase == "train":
            self.image_paths = input['P1_path'][0].split('.')[0] + '__'+ input['P2_path'][0].split('.')[0]
        else:
            self.image_paths = input['P1_path'][0].split('.')[0] + '__' + input['P2_path'][0].replace('.npy','')



    def forward(self):
        self.input_P1 = Variable(self.input_P1_set)
        self.input_BP1 = Variable(self.input_BP1_set)

        self.input_P2 = Variable(self.input_P2_set)
        self.input_BP2 = Variable(self.input_BP2_set)

        
        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1)]

        self.fake_p2, self.bg_fill, self.mask_2,self.out_patn_img,self.target_img_bg,self.target_img_fg = self.netG(G_input)


    def test(self):
        self.input_P1 = Variable(self.input_P1_set)
        self.input_BP1 = Variable(self.input_BP1_set)

        self.input_P2 = Variable(self.input_P2_set)
        self.input_BP2 = Variable(self.input_BP2_set)

        G_input = [self.input_P1,
                   torch.cat((self.input_BP1, self.input_BP2), 1)]
        self.fake_p2, self.bg_fill, self.mask_2,self.out_patn_img,self.target_img_bg,self.target_img_fg = self.netG(G_input)

    # get image paths
    def get_image_paths(self):
        return self.image_paths


    def backward_G(self):
        if self.opt.with_D_PB:
            pred_fake_PB = self.netD_PB(torch.cat((self.fake_p2, self.input_BP2), 1))
            self.loss_G_GAN_PB = self.criterionGAN(pred_fake_PB, True)

        if self.opt.with_D_PP:
            pred_fake_PP = self.netD_PP(torch.cat((self.fake_p2, self.input_P1), 1))
            self.loss_G_GAN_PP = self.criterionGAN(pred_fake_PP, True)


        # L1 loss
        
        if self.opt.L1_type == 'l1_plus_perL1' :
            losses = self.criterionL1(self.fake_p2, self.input_P2)
            self.loss_G_L1 = losses[0]
            self.loss_originL1 = losses[1].data
            self.loss_perceptual = losses[2].data
        
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_p2, self.input_P2) * self.opt.lambda_A
        pair_L1loss = self.loss_G_L1

        self.mask_target_hard = self.mask_2>0.95
        self.mask_target_hard = self.mask_target_hard.float() 
        self.loss_l1_fg = self.loss_l1(self.out_patn_img, (self.mask_target_hard*self.input_P2)) 
        loss_fg = self.loss_l1_fg

        self.mask_target_hard_bg = (1.0-self.mask_target_hard) 
        self.hard_bg = self.bg_fill*self.mask_target_hard_bg
        self.loss_l1_bg = self.loss_l1(self.hard_bg, self.input_P2*self.mask_target_hard_bg)
        loss_bg = self.loss_l1_bg
        

		
        if self.opt.with_D_PB:
            pair_GANloss = self.loss_G_GAN_PB * self.opt.lambda_GAN
            if self.opt.with_D_PP:
                pair_GANloss += self.loss_G_GAN_PP * self.opt.lambda_GAN
                pair_GANloss = pair_GANloss / 2
        else:
            if self.opt.with_D_PP:
                pair_GANloss = self.loss_G_GAN_PP * self.opt.lambda_GAN

        # print(pair_L1loss, pair_L1loss.shape, "pair_L1loss.shape................")

        if self.opt.with_D_PB or self.opt.with_D_PP:
            pair_loss =  pair_GANloss + loss_fg*5 + loss_bg*5 + pair_L1loss
            
        else:
            pair_loss = pair_L1loss

        pair_loss.backward()

        self.pair_loss = pair_loss.data
        if self.opt.with_D_PB or self.opt.with_D_PP:
            self.pair_GANloss = pair_GANloss.data


    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.lambda_GAN
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False) * self.opt.lambda_GAN
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    # D: take(P, B) as input
    def backward_D_PB(self):
        real_PB = torch.cat((self.input_P2, self.input_BP2), 1)
        # fake_PB = self.fake_PB_pool.query(torch.cat((self.fake_p2, self.input_BP2), 1))
        fake_PB = self.fake_PB_pool.query( torch.cat((self.fake_p2, self.input_BP2), 1).data )
        loss_D_PB = self.backward_D_basic(self.netD_PB, real_PB, fake_PB)
        self.loss_D_PB = loss_D_PB.data

    # D: take(P, P') as input
    def backward_D_PP(self):
        real_PP = torch.cat((self.input_P2, self.input_P1), 1)
        # fake_PP = self.fake_PP_pool.query(torch.cat((self.fake_p2, self.input_P1), 1))
        fake_PP = self.fake_PP_pool.query( torch.cat((self.fake_p2, self.input_P1), 1).data )
        loss_D_PP = self.backward_D_basic(self.netD_PP, real_PP, fake_PP)
        self.loss_D_PP = loss_D_PP.data


    def optimize_parameters(self):
        # forward
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D_P
        if self.opt.with_D_PP:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PP.zero_grad()
                self.backward_D_PP()
                self.optimizer_D_PP.step()

        # D_BP
        if self.opt.with_D_PB:
            for i in range(self.opt.DG_ratio):
                self.optimizer_D_PB.zero_grad()
                self.backward_D_PB()
                self.optimizer_D_PB.step()


    def get_current_errors(self):
        ret_errors = OrderedDict([ ('pair_loss', self.pair_loss)])
        # ret_errors = OrderedDict()
        if self.opt.with_D_PP:
            ret_errors['D_PP'] = self.loss_D_PP
        if self.opt.with_D_PB:
            ret_errors['D_PB'] = self.loss_D_PB
        if self.opt.with_D_PB or self.opt.with_D_PP:
            ret_errors['pair_GANloss'] = self.pair_GANloss

        # if self.opt.L1_type == 'l1_plus_perL1':
        #     ret_errors['origin_L1'] = self.loss_originL1
        #     ret_errors['perceptual'] = self.loss_perceptual

        return ret_errors

    def get_current_visuals(self):

        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)
        input_P2 = util.tensor2im(self.input_P2.data)
        bg_img_fill_ = util.tensor2im(self.bg_fill.data)
        out_patn_img_ = util.tensor2im(self.out_patn_img.data)
        target_img_bg_ = util.tensor2im(self.target_img_bg.data)
        target_img_fg_ = util.tensor2im(self.target_img_fg.data)
        target_img_hard_fg = util.tensor2im((self.mask_target_hard*self.input_P2).data)

        input_BP1 = util.draw_pose_from_map(self.input_BP1.data)[0]
        input_BP2 = util.draw_pose_from_map(self.input_BP2.data)[0]

        fake_p2 = util.tensor2im(self.fake_p2.data)
        mask_2 = (self.mask_2[0, :, :, :]).detach().cpu().numpy()
        mask_2_orig = np.transpose(mask_2,(1,2,0))
        mask_2 = ((np.transpose(mask_2,(1,2,0)))*255).astype(np.uint8())

        mask_hard = (self.mask_target_hard[0, :, :, :]).detach().cpu().numpy()
    	mask_hard = ((np.transpose(mask_hard,(1,2,0)))*255).astype(np.uint8())


    	
        vis = np.zeros((height, width*12, 3)).astype(np.uint8) #h, w, c
        vis[:, :width, :] = input_P1
        vis[:, width:width*2, :] = input_BP1
        vis[:, width*2:width*3, :] = input_P2
        vis[:, width*3:width*4, :] = input_BP2
        vis[:, width*4:width*5, :] = fake_p2
        vis[:, width * 5:width * 6, :] = bg_img_fill_
        vis[:, width * 6:width * 7, :] = out_patn_img_
        vis[:, width * 7:width * 8, :] = mask_2
        vis[:, width * 8:width * 9, :] = target_img_bg_
        vis[:, width * 9:width * 10, :] = target_img_fg_
        vis[:, width * 10:width * 11, :] = target_img_hard_fg
        vis[:, width * 11:width * 12, :] = mask_hard
        
        ret_visuals = OrderedDict([('vis', vis)])

        return ret_visuals

    def get_current_visuals_test(self):
        height, width = self.input_P1.size(2), self.input_P1.size(3)
        input_P1 = util.tensor2im(self.input_P1.data)

        input_BP1 = util.draw_pose_from_map(self.input_BP1.data)[0]
        input_BP2 = util.draw_pose_from_map(self.input_BP2.data)[0]

        bg_img_fill_ = util.tensor2im(self.bg_fill.data)
        out_patn_img_ = util.tensor2im(self.out_patn_img.data)
        
        target_img_fg_ = util.tensor2im(self.target_img_fg.data)
        fake_p2 = util.tensor2im(self.fake_p2.data)
        mask_2 = (self.mask_2[0, :, :, :]).detach().cpu().numpy()
        mask_2_orig = np.transpose(mask_2,(1,2,0))
        mask_2 = ((np.transpose(mask_2,(1,2,0)))*255).astype(np.uint8())
    	
        vis = np.zeros((height, width*7, 3)).astype(np.uint8) #h, w, c
        vis[:, :width, :] = input_P1
        vis[:, width:width*2, :] = input_BP1
        vis[:, width*2:width*3, :] = input_BP2
        vis[:, width*3:width*4, :] = bg_img_fill_
        vis[:, width*4:width*5, :] = out_patn_img_
        vis[:, width * 5:width * 6, :] = mask_2
        vis[:, width * 6:width * 7, :] = fake_p2  
        ret_visuals = OrderedDict([('vis', vis),('image', fake_p2), ('FG', target_img_fg_), ('mask', mask_2_orig)])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG,  'netG',  label, self.gpu_ids)
        if self.opt.with_D_PB:
            self.save_network(self.netD_PB,  'netD_PB',  label, self.gpu_ids)
        if self.opt.with_D_PP:
            self.save_network(self.netD_PP, 'netD_PP', label, self.gpu_ids)

