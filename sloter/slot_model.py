import torch
import torch.nn as nn
import torch.nn.functional as F
from sloter.utils.slot_attention import SlotAttention
from sloter.utils.position_encode import build_position_encoding
from timm.models import create_model
from collections import OrderedDict


class Identical(nn.Module):
    def __init__(self):
        super(Identical, self).__init__()

    def forward(self, x):
        return x


def load_backbone(args):
    bone = create_model(
        args.model,
        pretrained=args.pre_trained,
        num_classes=args.num_classes)
    if args.dataset == "MNIST":
        bone.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)
    if args.use_slot:
        if args.use_pre:
            checkpoint = torch.load(f"saved_model/{args.dataset}_no_slot_checkpoint.pth")
            new_state_dict = OrderedDict()
            for k, v in checkpoint["model"].items():
                name = k[9:] # remove `backbone.`
                new_state_dict[name] = v
            bone.load_state_dict(new_state_dict)
            print("load pre dataset parameter over")
        bone.global_pool = Identical()
        bone.fc = Identical()
    return bone


class SlotModel(nn.Module):
    def __init__(self, args):
        super(SlotModel, self).__init__()
        self.use_slot = args.use_slot
        self.backbone = load_backbone(args)
        if self.use_slot:
            self.channel = args.channel
            self.slots_per_class = args.slots_per_class
            self.conv1x1 = nn.Conv2d(self.channel, args.hidden_dim, kernel_size=(1, 1), stride=(1, 1))
            if args.pre_trained:
                self.dfs_freeze(self.backbone)
            self.slot = SlotAttention(args.num_classes, self.slots_per_class, args.hidden_dim, vis=args.vis,
                                         vis_id=args.vis_id, loss_status=args.loss_status, power=args.power, to_k_layer=args.to_k_layer)
            self.position_emb = build_position_encoding('sine', hidden_dim=args.hidden_dim)
            self.lambda_value = float(args.lambda_value)

    def dfs_freeze(self, model):
        for name, child in model.named_children():
            if "layer3" in name or "layer4" in name:
                continue
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

    def forward(self, x, target=None):
        x = self.backbone(x)
        if self.use_slot:
            x = self.conv1x1(x.view(x.size(0), self.channel, 9, 9))
            x = torch.relu(x)
            pe = self.position_emb(x)
            x_pe = x + pe

            b, n, r, c = x.shape
            x = x.reshape((b, n, -1)).permute((0, 2, 1))
            x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
            x, attn_loss = self.slot(x_pe, x)
        output = F.log_softmax(x, dim=1)

        if target is not None:
            if self.use_slot:
                loss = F.nll_loss(output, target) + self.lambda_value * attn_loss
            else:
                loss = F.nll_loss(output, target)
            return [output, loss]

        return output


# def get_args_parser():
#     parser = argparse.ArgumentParser('Set bird model', add_help=False)
#     parser.add_argument('--dataset_dir', default='/home/wbw/PAN/bird_200/CUB_200_2011',
#                         help='path for save data')
#     parser.add_argument('--model', default="efficientnet_b2", type=str)
#
#     # training set
#     parser.add_argument('--lr', default=0.0001, type=float)
#     parser.add_argument('--lr_drop', default=100, type=int)
#     parser.add_argument('--batch_size', default=16, type=int)
#     parser.add_argument('--weight_decay', default=0.0001, type=float)
#     parser.add_argument('--data_mode', default="rgb", type=str)
#     parser.add_argument('--epochs', default=200, type=int)
#     parser.add_argument("--num_classes", default=100, type=int)
#     parser.add_argument('--img_size', default=260, help='path for save data')
#     parser.add_argument('--pre_trained', default=True, help='whether use pre parameter for backbone')
#
#     # slot setting
#     parser.add_argument('--slots_per_class', default=1, help='number of slot for each class')
#     parser.add_argument('--vis', default=False, help='whether save slot visualization')
#     parser.add_argument('--vis_id', default=0, help='choose image to visualization')
#     return parser
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
#     args = parser.parse_args()
#     model = SlotModel(args)
#     print(model)