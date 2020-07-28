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


def test(model, device, test_loader, vis_id):
    test_loss = 0
    correct = 0
    for i_batch, sample_batch in enumerate(test_loader):
        model.eval()
        data = sample_batch["image"].to(device, dtype=torch.float32)
        target = sample_batch["label"].to(device, dtype=torch.int64)
        path = sample_batch["path"]
        output, loss = model(data, target)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        print(output[0])
        correct += pred.eq(target.view_as(pred)).sum().item()

        #For vis
        print(path[0])
        image_raw = Image.open(path[0]).resize((260,260))
        image_raw.save('sloter/vis/image.png')
        print(torch.argmax(output[vis_id]).item())
        model.train()
        # grad_cam = GradCam(model, target_layer='conv2', cam_extractor=CamExtractor)
        trans = transforms.ToTensor()

        for id in range(100):
            image_raw = Image.open('sloter/vis/image.png').convert('RGB')
            # image_raw_cam = Image.open('vis/image.png')
            slot_image = np.array(Image.open(f'sloter/vis/slot_{id}.png').resize(image_raw.size, resample=Image.BILINEAR), dtype=np.uint8)

            heatmap_only, heatmap_on_image = apply_colormap_on_image(image_raw, slot_image, 'gist_rainbow_r')
            heatmap_on_image.save(f'sloter/vis/slot_mask_{id}.png')

            # if id < 10:
            #     cam = grad_cam.generate_cam(trans(image_raw_cam).unsqueeze(1).cuda(), id)
            #     save_class_activation_images(image_raw, cam, f'{id}')

        break
        #################

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--slots', type=int, default=2, metavar='N',
                        help=' (default: 10)')
    parser.add_argument('--slots_per_class', type=int, default=5, metavar='N',
                        help=' (default: 1)')
    parser.add_argument('--use_slot', type=bool, default=True, metavar='N',
                        help=' (default: True)')
    parser.add_argument('--vis_id', type=int, default=0, metavar='N',
                        help=' (default: True)')
    parser.add_argument('--dataset_dir', default='/home/wbw/PAN/bird_200/CUB_200_2011/',
                        help='path for save data')
    parser.add_argument('--img_size', default=260, help='path for save data')
    parser.add_argument('--model', default="efficientnet_b2_slot", type=str)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True},
                      )

    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
    test_loader = torch.utils.data.DataLoader(bird.CUB_200(args, train=False, transform=bird.make_video_transform("val")), shuffle=False, **kwargs)

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=100)
    model.to(device)
    # Map model to be loaded to specified single gpu.
    checkpoint = torch.load("saved_model/checkpoint0119.pth")
    # new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        print(k)
    model.load_state_dict(checkpoint["model"])

    test(model, device, test_loader, vis_id=args.vis_id)


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


