import torch
from image_aug import ImageAugment
import torchvision.transforms.functional as F
import numpy as np


class Aug(object):
    """class for preprocessing images. """
    def __call__(self, image):
        wbw = ImageAugment()   # ImageAugment class will augment the img and label at same time
        seq = wbw.aug_sequence()
        image_aug = wbw.aug(image, seq)
        return image_aug


class ToTensor(object):
    """change sample to tensor"""
    def __init__(self, demo=False):
        self.demo = demo

    def __call__(self, image, color=True):
        image = torch.from_numpy((image/255).transpose([2, 0, 1]))  # convert numpy data to tensor
        return image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, imgs):
        imgs = F.normalize(imgs, mean=self.mean, std=self.std)
        return imgs