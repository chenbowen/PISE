import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.networks.base_network import BaseNetwork
from model.networks.base_function import *
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm

from util.util import feature_normalize

class ParsingNet(nn.Module):
    """
    define a parsing net to generate target parsing
    """
    def __init__(self, input_nc, output_nc, ngf=32, norm_layer=nn.InstanceNorm2d, act=nn.LeakyReLU(0.2), use_spect=False):
        super(ParsingNet, self).__init__()

        self.conv1 = BlockEncoder(input_nc, ngf*2, ngf, norm_layer, act, use_spect)
        self.conv2 = BlockEncoder(ngf*2, ngf*4, ngf*4, norm_layer, act, use_spect)

        self.conv3 = BlockEncoder(ngf*4, ngf*8, ngf*8, norm_layer, act, use_spect)
        self.conv4 = BlockEncoder(ngf*8, ngf*16, ngf*16, norm_layer, act, use_spect)
        self.deform3 = Gated_conv(ngf*16, ngf*16, norm_layer=norm_layer)
        self.deform4 = Gated_conv(ngf*16, ngf*16, norm_layer=norm_layer)

        self.up1 = ResBlockDecoder(ngf*16, ngf*8, ngf*8, norm_layer, act, use_spect)
        self.up2 = ResBlockDecoder(ngf*8, ngf*4, ngf*4, norm_layer, act, use_spect)


        self.up3 = ResBlockDecoder(ngf*4, ngf*2, ngf*2, norm_layer, act, use_spect)
        self.up4 = ResBlockDecoder(ngf*2, ngf, ngf, norm_layer, act, use_spect)

        self.parout = Output(ngf, 8, 3, norm_layer ,act, None)
        self.makout = Output(ngf, 1, 3, norm_layer, act, None)

    def forward(self, input):
        #print(input.shape)
        x = self.conv2(self.conv1(input))
        x = self.conv4(self.conv3(x))
        x = self.deform4(self.deform3(x))

        x = self.up2(self.up1(x))
        x = self.up4(self.up3(x))

        #print(x.shape)
        par = self.parout(x)
        mak = self.makout(x)
        
        par = (par + 1.) / 2.
        
        return par, mak


class PoseGenerator(BaseNetwork):
    def __init__(self, image_nc=3, structure_nc=18, output_nc=3, ngf=64, norm='instance', 
                activation='LeakyReLU', use_spect=True, use_coord=False):
        super(PoseGenerator, self).__init__()


        self.use_coordconv = True
        self.match_kernel = 3

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)
        
        self.parnet = ParsingNet(8+18*2, 8)

        self.Zencoder = Zencoder(3, ngf)

        self.imgenc = VggEncoder()
        self.getMatrix = GetMatrix(ngf*4, 1)
        
        self.phi = nn.Conv2d(in_channels=ngf*4+3, out_channels=ngf*4, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=ngf*4+3, out_channels=ngf*4, kernel_size=1, stride=1, padding=0)

        self.parenc = HardEncoder(8+18+8+3, ngf)

        self.dec = BasicDecoder(3)

        self.efb = EFB(ngf*4, 256)
        self.res = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)
                
        self.res1 = ResBlock(ngf*4, output_nc=ngf*4, hidden_nc=ngf*4, norm_layer=norm_layer, nonlinearity= nonlinearity,
                learnable_shortcut=False, use_spect=False, use_coord=False)

        self.loss_fn = torch.nn.MSELoss()

    def addcoords(self, x):
        bs, _, h, w = x.shape
        xx_ones = torch.ones([bs, h, 1], dtype=x.dtype, device=x.device)
        xx_range = torch.arange(w, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(1)
        xx_channel = torch.matmul(xx_ones, xx_range).unsqueeze(1)

        yy_ones = torch.ones([bs, 1, w], dtype=x.dtype, device=x.device)
        yy_range = torch.arange(h, dtype=x.dtype, device=x.device).unsqueeze(0).repeat([bs, 1]).unsqueeze(-1)
        yy_channel = torch.matmul(yy_range, yy_ones).unsqueeze(1)

        xx_channel = xx_channel.float() / (w - 1)
        yy_channel = yy_channel.float() / (h - 1)
        xx_channel = 2 * xx_channel - 1
        yy_channel = 2 * yy_channel - 1

        rr_channel = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
        
        concat = torch.cat((x, xx_channel, yy_channel, rr_channel), dim=1)
        return concat

    def computecorrespondence(self, fea1, fea2, temperature=0.01,
                detach_flag=False, WTA_scale_weight=1, alpha=1):
        ## normalize the feature, https://github.com/microsoft/CoCosNet
        def normalize(x):
            x = F.unfold(x, kernel_size=self.match_kernel, padding=int(self.match_kernel // 2))
            x = x - x.mean(dim=1, keepdim=True)
            x_norm = torch.norm(x, 2, 1, keepdim=True) + sys.float_info.epsilon
            x = torch.div(x, x_norm)
        
        theta = normalize(self.theta(fea1)).permute(0, 2, 1)
        phi = normalize(self.phi(fea2))

        f_WTA = torch.matmul(theta, phi) / temperature
        return F.softmax(f_WTA.permute(0,2,1), dim=-1)

    def forward(self, img1, img2, pose1, pose2, par1, par2):
        style_codes, exist_vector, img1code = self.Zencoder(img1, par1)
        ######### my par   for image editing.
        '''
        parcode,mask = self.parnet(torch.cat((par1, pose1, pose2),1))
        parsav = parcode
        par = torch.argmax(parcode, dim=1, keepdim=True)
        bs, _, h, w = par.shape
       # print(SPL2_img.shape,SPL1_img.shape)
        num_class = 8
        tmp = par.view( -1).long()
        ones = torch.sparse.torch.eye(num_class).cuda() 
        ones = ones.index_select(0, tmp)
        SPL2_onehot = ones.view([bs, h,w, num_class])
        #print(SPL2_onehot.shape)
        SPL2_onehot = SPL2_onehot.permute(0, 3, 1, 2)
        par2 = SPL2_onehot
        '''
        par2, mask = self.parnet(torch.cat((par1, pose1, pose2), 1))
        
        feature = self.parenc(torch.cat((par1, par2, pose2, img1), 1))
        
        # instance transfer, share weights to normalize features use efb prograssively
        parcode = self.efb(feature, par2, style_codes, exist_vector)
        parcode = self.res(parcode)

        ## regularization to let transformed code and target image code in the same feature space
        img2code = self.imgenc(img2)
        loss_reg = F.mse_loss(img2code, parcode)

        img1code = self.imgenc(img1)
        
        parcode1 = feature_normalize(parcode)
        img1code1 = feature_normalize(img1code)
        
        if self.use_coordconv:
            parcode1 = self.addcoords(parcode1)
            img1code1 = self.addcoords(img1code1)

        gamma, beta = self.getMatrix(img1code)
        bs, _, h, w = gamma.shape
        gamma = gamma.view(bs, 1, -1)
        beta = beta.view(bs, 1, -1)
        att = self.computecorrespondence(parcode1, img1code1)
        imgamma = torch.bmm(gamma, att).view(bs, 1, h, w).contiguous()
        imbeta = torch.bmm(beta, att).view(bs, 1, h, w).contiguous()

        parcode = parcode * (1 + imgamma) + imbeta 
        parcode = self.res1(parcode)

        pred_img = self.dec(parcode)
        return pred_img, loss_reg, par2







