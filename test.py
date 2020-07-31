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
from sloter.slot_model import load_model
from train import get_args_parser
from tools.data_loader import make_video_transform

from torchvision import datasets, transforms

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
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    model_name = f"{args.dataset}_use_slot_negative_checkpoint.pth"
    args.use_pre = False
    if "negative" in model_name:
        args.loss_status = -1
    else:
        args.loss_status = 1

    device = torch.device(args.device)
    
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        ])
    dataset_val = datasets.MNIST('./data/mnist', train=False, transform=transform)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
    image = iter(data_loader_val).next()[0]
    image_orl = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8)[0,0], mode='L')
    # image_path = os.path.join(args.dataset_dir, "images", "024.Red_faced_Cormorant", "Red_Faced_Cormorant_0007_796280.jpg")
    # image_orl = Image.open(image_path).convert('RGB')
    # image = np.array(image_orl.resize((args.img_size, args.img_size), Image.BILINEAR))
    # image = make_video_transform("val")(image)
    transform = transforms.Compose([
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    image = transform(image[0])

    model = load_model(args)
    # Map model to be loaded to specified single gpu.
    checkpoint = torch.load("saved_model/" + model_name, map_location="cuda:0")
    # new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        print(k)
    model.load_state_dict(checkpoint["model"])

    test(args, model, device, image_orl, image, vis_id=args.vis_id)


if __name__ == '__main__':
    main()


