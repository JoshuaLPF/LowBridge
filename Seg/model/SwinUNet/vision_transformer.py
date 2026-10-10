# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import torch
import torch.nn as nn
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=1000, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config
        self.swin_unet = SwinTransformerSys(img_size=img_size,
                                patch_size=4,
                                in_chans=3,
                                num_classes=self.num_classes,
                                embed_dim=96,  # 96  128
                                depths=[2, 2, 6, 2],  # [2, 2, 6, 2]  [2, 2, 18, 2]
                                num_heads=[3, 6, 12, 24],  # [3, 6, 12, 24]  [4, 8, 16, 32]
                                window_size=7,  # 7  12
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=True)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits

    def load_pre(self, pre_model):
        self.swin_unet.load_state_dict(torch.load(pre_model)['model'],strict=True)  #False
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
 