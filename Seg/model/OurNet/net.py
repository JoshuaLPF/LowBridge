import torch
import torch.nn as nn
import torch.nn.functional as F
# from ResNet import Backbone_ResNet50
from model.OurNet.dfformer import cdfformer_m36
from thop import profile
from thop import clever_format

def Conv3(in_dim, out_dim, stride=1, has_bias=False):
    "3x3 depth-wise convolution with padding"
    return nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=has_bias)

def Conv3_BN_ReLU(in_dim, out_dim, stride=1):
    return nn.Sequential(
            Conv3(in_dim, out_dim, stride),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),)

class OurNet(nn.Module):
    def __init__(self):
        super(OurNet, self).__init__()
        self.encoder = cdfformer_m36()
        # decoder
        self.upsample1 = nn.Sequential(Conv3_BN_ReLU(360, 180),
                                       Conv3_BN_ReLU(180, 180),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample2 = nn.Sequential(Conv3_BN_ReLU(528, 264),
                                       Conv3_BN_ReLU(264, 264),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample3 = nn.Sequential(Conv3_BN_ReLU(672, 336),
                                       Conv3_BN_ReLU(336, 336),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.upsample4 = nn.Sequential(Conv3_BN_ReLU(576, 288),
                                       Conv3_BN_ReLU(288, 288),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ))
        self.predictor = nn.Sequential(Conv3_BN_ReLU(180, 30),
                                       nn.UpsamplingBilinear2d(scale_factor=2, ),
                                       Conv3_BN_ReLU(30, 2))
    def forward(self, x):
        r = self.encoder(x)
        r1 = r[0] # 1,96,128,128     s18/36: 64    m36:96    b36:128          input_size 384: 96,48,24,12
        r2 = r[1] # 1,192,64,64      s18/36: 128   m36:192   b36:256          input_size 224: 56,28,14,7
        r3 = r[2] # 1,384,32,32      s18/36: 320   m36:384   b36:512          input_size 512: 128,64,32,16
        r4 = r[3] # 1,768,16,16      s18/36: 512   m36:576   b36:768
        # Decoder
        F4 = self.upsample4(r4)
        F3 = torch.cat((r3, F4), dim=1)
        F3 = self.upsample3(F3)
        F2 = torch.cat((r2, F3), dim=1)
        F2 = self.upsample2(F2)
        F1 = torch.cat((r1, F2), dim=1)
        F1 = self.upsample1(F1)
        out = self.predictor(F1)
        return out


if __name__ == '__main__':
    a = torch.randn(1, 3, 384, 384)
    c = torch.Tensor(a).cuda()
    model = OurNet().cuda()
    e, = model(c)