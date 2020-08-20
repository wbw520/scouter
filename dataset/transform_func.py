import torch
from tools.image_aug import ImageAugment
import torchvision.transforms.functional as F
from collections.abc import Sequence, Iterable
import numpy as np
from PIL import Image


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class Resize(object):
    """class for resize images. """
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        return np.array(F.resize(image, self.size, self.interpolation))

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class Aug(object):
    """class for preprocessing images. """
    def __init__(self, aug):
        self.aug = aug

    def __call__(self, image):
        if self.aug:
            ImgAug = ImageAugment()   # ImageAugment class will augment the img and label at same time
            seq = ImgAug.aug_sequence()
            image_aug = ImgAug.aug(image, seq)
            return image_aug
        else:
            return image

    def __repr__(self):
        return self.__class__.__name__ + 'Augmentation function'


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, image, color=True):
        if image.ndim == 2:
            image = image[:, :, None]
        image = torch.from_numpy(((image/255).transpose([2, 0, 1])).copy())  # convert numpy data to tensor
        return image

    def __repr__(self):
        return self.__class__.__name__ + '()'


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


def make_transform(args, mode):
    normalize_value = {"MNIST": [[0.1307], [0.3081]],
                       "CUB200": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
                     "ConText": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
                       "ImageNet": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]}
    selected_norm = normalize_value[args.dataset]
    normalize = Compose([
        ToTensor(),
        Normalize(selected_norm[0], selected_norm[1])
    ])

    if mode == "train":
        return Compose([
            Resize((args.img_size, args.img_size)),
            Aug(args.aug),
            normalize,
        ]
        )
    if mode == "val":
        return Compose([
            Resize((args.img_size, args.img_size)),
            normalize,
        ]
        )
    raise ValueError(f'unknown {mode}')