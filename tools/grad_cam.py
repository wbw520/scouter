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

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (args.img_size, args.img_size))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def image_deal(data_name):
    image_orl = Image.open(data_name).convert('RGB')
    image = image_orl.resize((args.img_size, args.img_size), Image.BILINEAR)
    image = make_transform(args, "val")(image)
    inputs = image.to(device, dtype=torch.float32)
    return inputs, image_orl


def show_cam_on_image(imgs, masks):
    final = np.uint8(255*masks)
    h, w, c = imgs.shape
    final = cv2.resize(final, (w, h), interpolation=cv2.INTER_LINEAR)
    final = cv2.applyColorMap(final, 2)
    final = final[:, :, (2, 1, 0)]
    out = cv2.addWeighted(np.uint8(imgs), 1.0, final, 0.3, 0)
    # ret, thresh = cv2.threshold(final, C.heat_value, 255, cv2.THRESH_BINARY)
    plt.figure(figsize=(10, 10), facecolor="#FFFFFF")
    plt.imshow(out)
    plt.show()


def make_grad(model, image_inf):
    grad_cam = GradCam(model=model, target_layer_names=[need_layer], use_cuda=True)
    input, img_heat = image_deal(image_inf)
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(torch.unsqueeze(input, dim=0), target_index)
    show_cam_on_image(np.array(img_heat), mask)


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
    root = os.path.join(args.dataset_dir, "images", "024.Red_faced_Cormorant", "Red_Faced_Cormorant_0007_796280.jpg")
    make_grad(init_model, root)
