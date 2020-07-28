import torch
import torch.nn as nn
import torch.nn.functional as F
from .slot_attention import SlotAttention, PositionEmbeddingSine
from .position_encode import build_position_encoding
import numpy as np


class Mnist_Model(nn.Module):
    def __init__(self, vis=False, slots_num=10, slots_per_class=1):
        super(Mnist_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        self.slot = SlotAttention(slots_num, slots_per_class, 64, vis=vis)
        
        head_layers = nn.ModuleList([nn.Linear(64, 1)])
        all_heads = [head_layers for i in range(slots_num)]
        self.heads = nn.ModuleList(all_heads)

        self.position_emb = build_position_encoding('sine', hidden_dim=64)


    def forward(self, x, selected_classes=None):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        pe = self.position_emb(x)
        x_pe = x + pe

        b, n, r, c = x.shape
        x = x.reshape((b, n, -1)).permute((0,2,1))
        x_pe = x_pe.reshape((b, n, -1)).permute((0,2,1))
        x = self.slot(x_pe, x, selected_classes=selected_classes)#.reshape((b, -1))

        x_temp = []

        for head_id, head in enumerate(self.heads):
            head_x = x[:, head_id:head_id+1]
            for head_layer_id, head_layer in enumerate(head):
                head_x = head_layer(head_x)
            x_temp.append(head_x.reshape((b, -1)))
      
        x = torch.cat(x_temp, dim=1)

        output = F.log_softmax(x, dim=1)
        return output


# 3*3 convolutino
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Mnist_Resnet_Model(nn.Module):
    def __init__(self, vis=False, slots_num=10, slots_per_class=1, slot=True, pre_trained=False):
        super(Mnist_Resnet_Model, self).__init__()
        self.slots_per_class = slots_per_class

        block = ResidualBlock
        layers = [2, 2, 2, 2]
        
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        # self.avg_pool = nn.AvgPool2d(8)

        self.slot = slot

        if self.slot:
            if pre_trained:
                load_dict = torch.load('saved_models/mnist_resnet_base.pt')
                self.load_my_state_dict(load_dict)
                self.dfs_freeze(self)
                # self.dfs_freeze_bnorm(self)

            self.slot = SlotAttention(slots_num, slots_per_class, 64, vis=vis)
            
            ###### Temp
            # head_layers = nn.ModuleList([nn.Linear(64, 1)])
            # # head_layers = nn.ModuleList([nn.Conv2d(1, 8, 3, 1, 1), nn.Conv2d(8, 1, 3, 1, 1), nn.Conv2d(1, 1, 8, 1, 0)])
            # all_heads = [head_layers for i in range(slots_num)]
            # self.heads = nn.ModuleList(all_heads)
            ###########

            self.position_emb = build_position_encoding('learned', hidden_dim=64)
        else:
            self.fc = nn.Linear(3136, 10)

    def dfs_freeze(self, model):
        for child in model.children():
            for param in child.parameters():
                param.requires_grad = False
            self.dfs_freeze(child)

    def dfs_freeze_bnorm(self, model):
        for name, child in model.named_children():
            if 'bn' not in name:
                self.dfs_freeze_bnorm(child)
                continue
            for param in child.parameters():
                param.requires_grad = False
            self.dfs_freeze_bnorm(child)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def load_my_state_dict(self, state_dict):

        ### Download weights from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        own_state = self.state_dict()
        # print(own_state.keys())
        for name, param in state_dict.items():
            # pdb.set_trace()
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

    def forward(self, x, selected_classes=None):
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.avg_pool(out)
        x = out

        if self.slot:
            pe = self.position_emb(x)
            x_pe = x + pe

            b, n, r, c = x.shape
            x = x.reshape((b, n, -1)).permute((0,2,1))
            x_pe = x_pe.reshape((b, n, -1)).permute((0,2,1))
            x, attn_loss = self.slot(x_pe, x, selected_classes=selected_classes)#.reshape((b, -1))

            # x_temp = []

            # for head_id, head in enumerate(self.heads):
            #     head_x = x[:, head_id:head_id+1]#.reshape((x.size(0), 1, -1))
            #     for head_layer_id, head_layer in enumerate(head):
            #         head_x = head_layer(head_x)
            #     x_temp.append(head_x.reshape((b, -1)))
        
            # x = torch.cat(x_temp, dim=1)
        
        else:
            x = x.view((x.size(0), -1))
            x = self.fc(x)

        output = F.log_softmax(x, dim=1)
        return output, attn_loss