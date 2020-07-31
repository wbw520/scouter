from dataset.mnist import MNIST
from dataset.CUB200 import CUB_200
from dataset.transform_func import make_transform


def select_dataset(args):
    if args.dataset == "MNIST":
        dataset_train = MNIST('./data/mnist', train=True, download=True, transform=make_transform(args, "train"))
        dataset_val = MNIST('./data/mnist', train=False, transform=make_transform(args, "val"))
        return dataset_train, dataset_val
    if args.dataset == "CUB200":
        dataset_train = CUB_200(args, train=True, transform=make_transform(args, "train"))
        dataset_val = CUB_200(args, train=False, transform=make_transform(args, "val"))
        return dataset_train, dataset_val

    raise ValueError(f'unknown {args.dataset}')

