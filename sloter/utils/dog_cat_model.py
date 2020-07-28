import torch
import torch.nn as nn
import torch.nn.functional as F
from .slot_attention import SlotAttention, PositionEmbeddingSine
from .position_encode import build_position_encoding
import numpy as np
from torchvision.models import resnet18
# from .spinenet.spinenet import SpineNet


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        # print(x.shape)
        return x


class Model(nn.Module):
    def __init__(self, vis=False, vis_id=0, slots_num=10, slots_per_class=1, slot=True, pre_trained=False):
        super(Model, self).__init__()
        self.slots_per_class = slots_per_class

        self.backbone = resnet18(pretrained=True)
        # self.backbone.layer3 = Identity()
        # self.backbone.layer4 = Identity()
        self.backbone.fc = Identity()

        # self.backbone = SpineNet('49', output_level=[3,4,5,6,7])

        self.slot = slot

        if self.slot:
            self.conv1x1 = nn.Conv2d(512, 64, 1, 1, 0)
            self.backbone.avgpool = Identity()#nn.AdaptiveAvgPool2d(output_size=(32,32))
            if pre_trained:
                load_dict = torch.load('saved_models/dog_cat_base.pt')
                self.load_my_state_dict(load_dict)
                self.dfs_freeze(self)
                self.dfs_freeze_bnorm(self)

            self.slot = SlotAttention(slots_num, slots_per_class, 64, vis=vis, vis_id=vis_id)

            self.position_emb = build_position_encoding('learned', hidden_dim=64)
        else:
            self.fc = nn.Linear(512, 2)

    def dfs_freeze(self, model):
        for name, child in model.named_children():
            if 'layer3' in name or 'layer4' in name:
                # print(name,0)
                continue
            # print(name, 1)
            for param in child.parameters():
                param.requires_grad = False
            self.dfs_freeze(child)

    def dfs_freeze_bnorm(self, model):
        for name, child in model.named_children():
            if 'bn' not in name:
                self.dfs_freeze_bnorm(child)
                continue
            # print(name)
            for param in child.parameters():
                param.requires_grad = False
            self.dfs_freeze_bnorm(child)

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        # print(own_state.keys())
        for name, param in state_dict.items():
            # pdb.set_trace()
            name = name.replace('module.', '')
            if name in own_state.keys():
                # print(name)
                if isinstance(param, nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, '
                                       'whose dimensions in the model are {} and '
                                       'whose dimensions in the checkpoint are {}.'
                                       .format(name, own_state[name].size(), param.size()))
            else:
                print('NAME IS NOT IN OWN STATE::>' + name)

    def forward(self, x, target=None, selected_classes=None):
        x = self.backbone(x)
        if self.slot:
            x = x.view(x.size(0), 512, 16, 16)
            x = self.conv1x1(x)
            # pe = self.position_emb(x)
            # x_pe = x + pe

            b, n, r, c = x.shape
            x = x.reshape((b, n, -1)).permute((0,2,1))
            # x_pe = x_pe.reshape((b, n, -1)).permute((0,2,1))
            x, attn_loss = self.slot(x, x, selected_classes=selected_classes, target=target)#.reshape((b, -1))
            # x, attn_loss = self.slot(x_pe, x, selected_classes=selected_classes)#.reshape((b, -1))

        else:
            # x = x.view((x.size(0), -1))
            x = self.fc(x.view((x.size(0), -1)))
            attn_loss = 0

        output = F.log_softmax(x, dim=1)

        if target is not None:
            output = F.nll_loss(output, target) + attn_loss

        return output