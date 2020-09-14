import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import random
import numpy as np


class ImageAugment(object):
    """
    class for augment the training data using imgaug
    """
    def __init__(self):
        self.key = 0
        self.choice = 1
        self.rotate = np.random.randint(-10, 10)
        self.scale_x = random.uniform(0.8, 1.0)
        self.scale_y = random.uniform(0.8, 1.0)
        self.translate_x = random.uniform(0, 0.1)
        self.translate_y = random.uniform(-0.1, 0.1)
        self.brightness = np.random.randint(-10, 10)
        self.linear_contrast = random.uniform(0.5, 2.0)
        self.alpha = random.uniform(0, 1.0)
        self.lightness = random.uniform(0.75, 1.5)
        self.Gaussian = random.uniform(0.0, 0.05*255)
        self.Gaussian_blur = random.uniform(0, 3.0)

    def aug(self, image, sequence):
        """
        :param image: need size (H, W, C) one image once
        :param sequence: collection of augment function
        :return:
        """
        image_aug = sequence(image=image)
        return image_aug

    def rd(self, rand_max):
        seed = np.random.randint(0, rand_max)
        return seed

    def aug_sequence(self):
        sequence = self.aug_function()
        seq = iaa.Sequential(sequence, random_order=True)
        return seq

    def aug_function(self):
        sequence = []
        if self.rd(2) == self.key:
            sequence.append(iaa.Fliplr(1.0))  # 50% horizontally flip all batch images
        if self.rd(2) == self.key:
            sequence.append(iaa.Flipud(1.0))  # 50% vertically flip all batch images
        if self.rd(2) == self.key:
            sequence.append(iaa.Affine(
                scale={"x": self.scale_x, "y": self.scale_y},  # scale images to 80-100% of their size
                translate_percent={"x": self.translate_x, "y": self.translate_y},  # translate by -10 to +10 percent (per axis)
                rotate=(self.rotate),  # rotate by -15 to +15 degrees
            ))
        if self.rd(2) == self.key:
            sequence.extend(iaa.SomeOf((1, self.choice),
                                       [
                                           iaa.OneOf([
                                               iaa.GaussianBlur(self.Gaussian_blur),  # blur images with a sigma between 0 and 3.0
                                               # iaa.AverageBlur(k=(2, 7)),  # blur images using local means with kernel size 2-7
                                               # iaa.MedianBlur(k=(3, 11))  # blur images using local medians with kernel size 3-11
                                           ]),
                                           # iaa.Sharpen(alpha=self.alpha, lightness=self.lightness),  # sharpen images
                                           # iaa.LinearContrast(self.linear_contrast, per_channel=0.5),  # improve or worse the contrast
                                           # iaa.Add(self.brightness, per_channel=0.5),  # change brightness
                                           # iaa.AdditiveGaussianNoise(loc=0, scale=0.1, per_channel=0.5)  # add gaussian n
                                       ]))
        return sequence


def show_aug(image):
    plt.figure(figsize=(10, 10), facecolor="#FFFFFF")
    for i in range(1, len(image)+1):
        plt.subplot(len(image), 1, i)
        plt.imshow(image[i-1])
    plt.show()

