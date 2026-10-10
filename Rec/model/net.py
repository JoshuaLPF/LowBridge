import torch
import torch.nn as nn
from model.dfformer import cdfformer_m36

def Conv3(in_dim, out_dim, stride=1, has_bias=False):
    return nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1, bias=has_bias)

def Conv3_BN_ReLU(in_dim, out_dim, stride=1):
    return nn.Sequential(
            Conv3(in_dim, out_dim, stride),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),)

class OurNet(nn.Module):
    def __init__(self, pretrained_backbone=True):
        super().__init__()
        self.encoder = cdfformer_m36(pretrained=pretrained_backbone)
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
                                       Conv3_BN_ReLU(30, 1))
    def forward(self, x):
        r = self.encoder(x)
        r1, r2, r3, r4 = r
        F4 = self.upsample4(r4)
        F3 = torch.cat((r3, F4), dim=1)
        F3 = self.upsample3(F3)
        F2 = torch.cat((r2, F3), dim=1)
        F2 = self.upsample2(F2)
        F1 = torch.cat((r1, F2), dim=1)
        F1 = self.upsample1(F1)
        out = self.predictor(F1)
        return out
