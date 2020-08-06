import torch
from torch.autograd import Variable
from torch.nn import functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np
from train import get_args_parser
import argparse
from sloter.slot_model import load_backbone
from dataset.transform_func import make_transform
import os
from PIL import Image

from torchvision import datasets, transforms
from dataset.ConText import ConText, MakeList
from dataset.CUB200 import CUB_200
from sloter.utils.vis import apply_colormap_on_image

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            print(x.size())
            if name in "global_pool":
                batch = x.size()[0]  # drop height and width
                channel = x.size()[1]
                x = x.view(batch, channel)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = F.softmax(output, dim=1)
        return target_activations, output


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.to(device)

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        print("predict cat:", index)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = cv2.resize(cam, (args.img_size, args.img_size))
        return cam


def image_deal(data_name):
    image_orl = Image.open(data_name).convert('RGB')
    image = make_transform(args, "val")(image)
    inputs = image.to(device, dtype=torch.float32)
    return inputs, image_orl


def show_cam_on_image(img, masks, target_index):
    final = np.uint8(255*masks)

    heatmap_only, heatmap_on_image = apply_colormap_on_image(img, final, 'jet')
    heatmap_on_image.save(f'sloter/vis/grad_cam_{target_index}.png')


def make_grad(model, inputs, img_heat, grad_min_level):
    grad_cam = GradCam(model=model, target_layer_names=[need_layer], use_cuda=True)
    img_heat = img_heat.resize((args.img_size, args.img_size), Image.BILINEAR)
    # inputs, img_heat = image_deal(image_inf)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    # target_index = None
    for target_index in range(0, args.num_classes):
        mask = grad_cam(torch.unsqueeze(inputs, dim=0), target_index)
        mask = np.maximum(mask, 0)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        mask = np.maximum(mask, grad_min_level)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        show_cam_on_image(img_heat, mask, target_index)


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for model and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    need_layer = "layer4"
    device = args.device
    init_model = load_backbone(args)

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
        image = data["image"][98]#19 21  26  59  61 98 22*35 40*   41&
        label = data["label"][98]#19 21  26  59  61 98 22*35 40*   41&
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
        # image_path = os.path.join(args.dataset_dir, "images", "001.Black_footed_Albatross", "Black_Footed_Albatross_0001_796111.jpg")
        # image_orl = Image.open(image_path).convert('RGB')
        # image = transform(image_orl)
        # transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # label = ''
        dataset_val = CUB_200(args, train=False, transform=transform)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
        data = iter(data_loader_val).next()
        image = data["image"][6]
        label = data["label"][6]
        image_orl = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8).transpose((1,2,0)), mode='RGB')
        image = transform(image_orl)
        transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = transform(image)

    make_grad(init_model, image, image_orl, args.grad_min_level)
