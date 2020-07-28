from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from timm.models import create_model
from torch.utils.data import Dataset
import os, os.path
import tools.data_loader as bird
from collections import OrderedDict
from sloter.utils.vis import apply_colormap_on_image
from sloter.slot_model import SlotModel
from train import get_args_parser
from tools.data_loader import make_video_transform


def test(args, model, device, img, image, vis_id):
    model.to(device)
    model.eval()
    image = image.to(device, dtype=torch.float32)
    output = model(torch.unsqueeze(image, dim=0))
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    print(output[0])
    print(pred[0])

    #For vis
    image_raw = img
    image_raw.save('sloter/vis/image.png')
    print(torch.argmax(output[vis_id]).item())
    model.train()
    # grad_cam = GradCam(model, target_layer='conv2', cam_extractor=CamExtractor)
    trans = transforms.ToTensor()

    for id in range(args.num_classes):
        image_raw = Image.open('sloter/vis/image.png').convert('RGB')
        # image_raw_cam = Image.open('vis/image.png')
        slot_image = np.array(Image.open(f'sloter/vis/slot_{id}.png').resize(image_raw.size, resample=Image.BILINEAR), dtype=np.uint8)

        heatmap_only, heatmap_on_image = apply_colormap_on_image(image_raw, slot_image, 'gist_rainbow_r')
        heatmap_on_image.save(f'sloter/vis/slot_mask_{id}.png')

        # if id < 10:
        #     cam = grad_cam.generate_cam(trans(image_raw_cam).unsqueeze(1).cuda(), id)
        #     save_class_activation_images(image_raw, cam, f'{id}')


def main():
    # Training settings
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)

    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
    image_path = os.path.join(args.dataset_dir, "images", "024.Red_faced_Cormorant", "Red_Faced_Cormorant_0007_796280.jpg")
    image_orl = Image.open(image_path).convert('RGB')
    image = np.array(image_orl.resize((260, 260), Image.BILINEAR))
    image = make_video_transform("val")(image)

    model = SlotModel(args)
    # Map model to be loaded to specified single gpu.
    checkpoint = torch.load("saved_model/use_slot_negetive_checkpoint0149.pth", map_location="cuda:0")
    # new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        print(k)
    model.load_state_dict(checkpoint["model"])

    test(args, model, device, image_orl, image, vis_id=args.vis_id)


class DC_Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, for_train=True):
        self.transform = transform

        images = sorted(os.listdir(root_dir))
        self.images = [os.path.join(root_dir, image) for image in images]
        self.size = (512, 512)#Image.open(self.images[0]).size
        if for_train:
            self.images = self.images[:-2000]
        else:
            self.images = self.images[-2000:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).resize(self.size)
        label = 0 if 'dog' in os.path.basename(self.images[idx]) else 1

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    main()


