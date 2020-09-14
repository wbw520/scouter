import numpy as np
import torch
from sloter.slot_model import load_backbone
import argparse
from train import get_args_parser
from torchvision import datasets, transforms
from matplotlib.colors import LinearSegmentedColormap
import os
from PIL import Image
import torch.nn.functional as F
from sloter.utils.vis import apply_colormap_on_image
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    GuidedGradCam,
    LayerGradCam,
    LayerAttribution,
    LayerDeepLiftShap,
    LayerDeepLift
)
from tqdm import tqdm
from dataset.ConText import ConText, MakeList, MakeListImage
from dataset.CUB200 import CUB_200


def show_cam_on_image(img, masks, target_index, save_name):
    final = np.uint8(255*masks)

    mask_image = Image.fromarray(final, mode='L')
    mask_image.save(f'sloter/vis/{save_name}_{target_index}_mask.png')

    heatmap_only, heatmap_on_image = apply_colormap_on_image(img, final, 'jet')
    heatmap_on_image.save(f'sloter/vis/{save_name}_{target_index}.png')


def make_grad(attribute_f, inputs, img_heat, grad_min_level, save_name):
    img_heat = img_heat.resize((args.img_size, args.img_size), Image.BILINEAR)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    # target_index = None
    for target_index in tqdm(range(0, args.num_classes)):
        mask = attribute_f.attribute(inputs, target=target_index)
        if mask.size(1) > 1:
            mask = torch.mean(mask, dim=1, keepdim=True)
        mask = F.interpolate(mask, size=(args.img_size, args.img_size), mode="bilinear")
        mask = mask.squeeze(dim=0).squeeze(dim=0)
        mask = mask.detach().numpy()
        mask = np.maximum(mask, 0)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        mask = np.maximum(mask, grad_min_level)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        show_cam_on_image(img_heat, mask, target_index, save_name)


def for_vis(args):
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])
    # Con-text
    if args.dataset == 'ConText':
        train, val = MakeList(args).get_data()
        dataset_val = ConText(val, transform=transform)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
        data = iter(data_loader_val).next()
        image = data["image"][0]
        label = data["label"][0]
        image_orl = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8).transpose((1,2,0)), mode='RGB')
        image = transform(image_orl)
        transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif args.dataset == 'ImageNet':
        train, val = MakeListImage(args).get_data()
        dataset_val = ConText(val, transform=transform)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
        iter_loader = iter(data_loader_val)
        for i in range(0, 1):
            data = iter_loader.next()
        image = data["image"][0]
        label = data["label"][0].item()
        image_orl = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8).transpose((1,2,0)), mode='RGB')
        image = transform(image_orl)
        transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # MNIST
    elif args.dataset == 'MNIST':
        dataset_val = datasets.MNIST('./data/mnist', train=False, transform=transform)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
        image = iter(data_loader_val).next()[0][0]
        label = ''
        image_orl = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8)[0], mode='L')
        image = transform(image_orl)
        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
    # CUB
    elif args.dataset == 'CUB200':
        dataset_val = CUB_200(args, train=False, transform=transform)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
        data = iter(data_loader_val).next()
        image = data["image"][0]
        label = data["label"][0]
        image_orl = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8).transpose((1,2,0)), mode='RGB')
        image = transform(image_orl)
        transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = transform(image)
    image = image.unsqueeze(0)

    model = load_backbone(args)
    model.eval()
    output = model(image)
    output = F.softmax(output, dim=1)
    print(output)
    print(output.size())
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = str(pred_label_idx.item() + 1)
    print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

    # gradients = LayerGradCam(model, layer=model.layer4)
    # make_grad(gradients, image, image_orl, args.grad_min_level, 'GradCam')
    gradients = LayerDeepLift(model, layer=model.layer4)
    make_grad(gradients, image, image_orl, args.grad_min_level, 'DeepLIFT')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    args_dict = vars(args)
    args_for_evaluation = ['num_classes', 'lambda_value', 'power', 'slots_per_class']
    args_type = [int, float, int, int]
    for arg_id, arg in enumerate(args_for_evaluation):
        args_dict[arg] = args_type[arg_id](args_dict[arg])

    for_vis(args)
