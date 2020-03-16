import torch.nn as nn
import functools
import torch
import functools
import torch.nn.functional as F
from torch.autograd import Variable

class PATBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False):
        super(PATBlock, self).__init__()
        self.conv_block_stream1 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=False)
        self.conv_block_stream2 = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias, cal_att=True, cated_stream2=cated_stream2)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, cated_stream2=False, cal_att=False):
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

        if cated_stream2:
            conv_block += [nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim*2,track_running_stats=True),
                       nn.ReLU(True)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                           norm_layer(dim,track_running_stats=True),
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

        if cal_att:
            if cated_stream2:
                conv_block += [nn.Conv2d(dim*2, dim, kernel_size=3, padding=p, bias=use_bias)]
            else:
                conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)]
        else:
            conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim,track_running_stats=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x1, x2):
        x1_out = self.conv_block_stream1(x1)
        x2_out = self.conv_block_stream2(x2)
        att = F.sigmoid(x2_out)

        x1_out = x1_out * att
        out = x1 + x1_out # residual connection

        # stream2 receive feedback from stream1
        x2_out = torch.cat((x2_out, out), 1)
        return out, x2_out, x1_out


class PATNModel(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', n_downsampling=2):
        assert(n_blocks >= 0 and type(input_nc) == list)
        super(PATNModel, self).__init__()
        self.input_nc_s1 = input_nc[0]
        self.input_nc_s2 = input_nc[1]
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # down_sample
        model_stream1_down = [nn.ReflectionPad2d(3),
                    nn.Conv2d(self.input_nc_s1, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                    norm_layer(ngf,track_running_stats=True),
                    nn.ReLU(True)]

        model_stream2_down = [nn.ReflectionPad2d(3),
                    nn.Conv2d(self.input_nc_s2, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                    norm_layer(ngf,track_running_stats=True),
                    nn.ReLU(True)]

        # n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model_stream1_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf * mult * 2,track_running_stats=True),
                            nn.ReLU(True)]
            model_stream2_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                            norm_layer(ngf * mult * 2,track_running_stats=True),
                            nn.ReLU(True)]

        # att_block in place of res_block
        mult = 2**n_downsampling
        cated_stream2 = [True for i in range(n_blocks)]
        cated_stream2[0] = False
        attBlock = nn.ModuleList()
        for i in range(n_blocks):
            attBlock.append(PATBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, cated_stream2=cated_stream2[i]))

        # up_sample
        model_stream1_up = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_stream1_up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                            norm_layer(int(ngf * mult / 2),track_running_stats=True),
                            nn.ReLU(True)]


        model_stream1_up_end = []
        model_stream1_up_end += [nn.ReflectionPad2d(3)]
        model_stream1_up_end += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_stream1_up_end += [nn.Tanh()]

        model_out_mask = []
        model_out_mask += [nn.ReflectionPad2d(3)]
        model_out_mask += [nn.Conv2d(ngf, 1, kernel_size=7, padding=0)]
        model_out_mask += [nn.Sigmoid()]

        # self.model = nn.Sequential(*model)
        self.stream1_down = nn.Sequential(*model_stream1_down)
        self.stream2_down = nn.Sequential(*model_stream2_down)
        # self.att = nn.Sequential(*attBlock)
        self.att = attBlock
        self.stream1_up = nn.Sequential(*model_stream1_up)
        self.stream1_up_end = nn.Sequential(*model_stream1_up_end)
        self.out_mask = nn.Sequential(*model_out_mask)

    def forward(self, input): # x from stream 1 and stream 2
        # here x should be a tuple
        x1, x2 = input
        # down_sample
        x1 = self.stream1_down(x1)
        x2 = self.stream2_down(x2)
        # att_block
        for model in self.att:
            x1, x2, _ = model(x1, x2)

        # up_sample
        x1 = self.stream1_up(x1)
        out_image = self.stream1_up_end(x1)
        out_mask = self.out_mask(x1)
        return out_image, out_mask

class UnetMask(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, gpu_ids, padding_type, model):
        super(UnetMask, self).__init__()

        self.model = model
        #input img feature extraction input size(128,64)
        self.img_feat1_skip = nn.Conv2d(3,64,(7,7),stride=(1,1),padding=(3,3))
        self.batch_norm = nn.BatchNorm2d(64) 
        #downsample
        self.img_down1 = nn.Conv2d(64,64,(3,3),stride=(2,2),padding=(1,1))
        #input pose feature extraction
        self.pose_feat1 = nn.Conv2d(18,32,(7,7),stride=(1,1),padding=(3,3))
        #downsample
        self.pose_down1 = nn.Conv2d(32,32,(3,3),stride=(2,2),padding=(1,1))
        #cat img and pose and convolv
        self.img_pose_feat_skip2 = nn.Conv2d(96,96,(3,3),stride=(1,1),padding=(1,1))
        self.batch_norm2 = nn.BatchNorm2d(96)

        #downsample till size(8,4)
        self.down_2 = nn.Conv2d(96,128,(3,3),stride=(2,2),padding=(1,1))
        self.conv_3_skip3 = nn.Conv2d(128,128,(3,3),stride=(1,1),padding=(1,1))
        self.batch_norm3 = nn.BatchNorm2d(128)

        self.down_3 = nn.Conv2d(128,256,(3,3),stride=(2,2),padding=(1,1))
        self.conv_4_skip4 = nn.Conv2d(256,256,(3,3),stride=(1,1),padding=(1,1))
        self.batch_norm4 = nn.BatchNorm2d(256)

        self.down_4 = nn.Conv2d(256,512,(3,3),stride=(2,2),padding=(1,1))

        #bottleneck
        self.bottleneck = nn.Conv2d(512,512,(3,3),stride=(1,1),padding=(1,1))
        self.batch_norm_bn = nn.BatchNorm2d(512)

        #upsample 
        self.up_4 = nn.ConvTranspose2d(512,256,(3,3),stride=(2,2),padding=(1,1),output_padding=1)
        self.conv_cat_4 = nn.Conv2d(512,256,(3,3),stride=(1,1),padding=(1,1))
        self.batch_norm_up4 = nn.BatchNorm2d(256)

        self.up_3 = nn.ConvTranspose2d(256,128,(3,3),stride=(2,2),padding=(1,1),output_padding=1)
        self.conv_cat_3 = nn.Conv2d(256,128,(3,3),stride=(1,1),padding=(1,1))
        self.batch_norm_up3 = nn.BatchNorm2d(128)

        self.up_2 = nn.ConvTranspose2d(128,96,(3,3),stride=(2,2),padding=(1,1),output_padding=1)
        self.conv_cat_2 = nn.Conv2d(192,96,(3,3),stride=(1,1),padding=(1,1))
        self.batch_norm_up2 = nn.BatchNorm2d(96)

        self.up_1 = nn.ConvTranspose2d(96,64,(3,3),stride=(2,2),padding=(1,1),output_padding=1)
        self.conv_cat_1 = nn.Conv2d(128,64,(3,3),stride=(1,1),padding=(1,1))
        self.batch_norm_up1 = nn.BatchNorm2d(64)

        self.conv_feat_compress = nn.Conv2d(64,3,(7,7),stride=(1,1),padding=(3,3))

        #if the model is fg_mask
        self.final_conv = nn.Conv2d(3,1,(3,3),stride=(1,1),padding=(1,1))
        #if the model is bg_sythesis
        self.bg_final_conv = nn.Conv2d(3,3,(3,3),stride=(1,1),padding=(1,1))


    def forward(self,input):
        input_img = input[0]
        input_pose = input[1][:,0:18,:,:]
        #feat extract
        feat1_skip = self.batch_norm(F.relu(self.img_feat1_skip(input_img)))
        feat1_down = F.relu(self.img_down1(feat1_skip))
        feat_pose1 = F.relu(self.pose_feat1(input_pose))
        feat_pose_down = F.relu(self.pose_down1(feat_pose1))

        #downsampling 
        #cat img and pose feat.
        in2 = torch.cat((feat1_down,feat_pose_down), 1)
        #skip2
        out2_skip = self.batch_norm2(F.relu(self.img_pose_feat_skip2(in2)))
        out2_down = F.relu(self.down_2(out2_skip))
        #skip3
        out3_skip = self.batch_norm3(F.relu(self.conv_3_skip3(out2_down)))
        out3_down = F.relu(self.down_3(out3_skip))
        #skip4
        out4_skip = self.batch_norm4(F.relu(self.conv_4_skip4(out3_down)))
        out4_down = F.relu(self.down_4(out4_skip))
        
        bottleneck = self.batch_norm_bn(F.relu(self.bottleneck(out4_down)))
        out_up4 = F.relu(self.up_4(bottleneck))

        in_up4 = torch.cat((out_up4, out4_skip), 1)
        out_cat4 = self.batch_norm_up4(F.relu(self.conv_cat_4(in_up4)))
        out_up3 = F.relu(self.up_3(out_cat4))

        in_up3 = torch.cat((out_up3, out3_skip), 1)
        out_cat3 = self.batch_norm_up3(F.relu(self.conv_cat_3(in_up3)))
        out_up2 = F.relu(self.up_2(out_cat3))

        in_up2 = torch.cat((out_up2, out2_skip), 1)
        out_cat2 = self.batch_norm_up2(F.relu(self.conv_cat_2(in_up2)))
        out_up1 = F.relu(self.up_1(out_cat2))

        in_up1 = torch.cat((out_up1, feat1_skip), 1)
        out_cat1 = self.batch_norm_up1(F.relu(self.conv_cat_1(in_up1)))
        up_final = F.relu(self.conv_feat_compress(out_cat1))
        if self.model == "fg_mask":
            output = F.sigmoid(self.final_conv(up_final))
        else:
            output = F.tanh(self.bg_final_conv(up_final))
        return output


class PATNetwork(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', n_downsampling=2):
        super(PATNetwork, self).__init__()
        assert type(input_nc) == list and len(input_nc) == 2, 'The AttModule take input_nc in format of list only!!'
        self.gpu_ids = gpu_ids
        self.mask_model = UnetMask(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, gpu_ids, padding_type, model="fg_mask")
        self.model = PATNModel(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, gpu_ids, padding_type, n_downsampling=n_downsampling)
        self.inpaint_model = UnetMask(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, gpu_ids, padding_type, model="bg_synthesis")

    def forward(self, input):
        if self.gpu_ids and isinstance(input[0].data, torch.cuda.FloatTensor):
            fg_mask = nn.parallel.data_parallel(self.mask_model, input, self.gpu_ids)
            fg_img = fg_mask*input[0]
            in_patn = [fg_img, input[1]]
            in_bg = [(1-fg_mask)*input[0], input[1]]
            bg_fill = nn.parallel.data_parallel(self.inpaint_model, in_bg, self.gpu_ids)
            out_img, out_mask = nn.parallel.data_parallel(self.model, in_patn, self.gpu_ids)
            target_bg = bg_fill*(1-out_mask)
            target_image_fg = out_img*out_mask
            final_image = target_bg+target_image_fg 
            return final_image, fg_mask, fg_img, (1-fg_mask)*input[0], bg_fill, out_mask, out_img, target_bg, target_image_fg
        else:
            return self.model(input)






