from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import random
from utils.dog_cat_model import Model
from PIL import Image
import os, os.path
from torch.utils.data import Dataset


def train(args, model, device, train_loader, optimizer, epoch, slots_num):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # selected_classes = np.zeros((data.size(0), slots_num))
        # for b_id, b in enumerate(selected_classes):
        #     for t_id, t in enumerate(b):
        #         if target[b_id].item() == t_id:
        #             selected_classes[b_id, t_id] = 1
        #         else:
        #             selected_classes[b_id, t_id] = random.randint(0,1)
        selected_classes = None

        loss = model(data, target=target, selected_classes=selected_classes).mean()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return correct


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
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
    parser.add_argument('--pre_trained', type=bool, default=True, metavar='N',
                        help=' (default: False)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 32,
                       'pin_memory': True,},
                     )

    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    train_loader = torch.utils.data.DataLoader(DC_Dataset('../../../data/dogs-vs-cats/train', transform=transform), shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(DC_Dataset('../../../data/dogs-vs-cats/train', transform=transform, for_train=False), shuffle=False, **kwargs)

    model = Model(slots_num=args.slots, slots_per_class=args.slots_per_class, slot=args.use_slot, pre_trained=args.pre_trained).to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    best_correct = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, args.slots)
        correct = test(model, device, test_loader)
        scheduler.step()
        print(correct, best_correct)
        if True or correct >= best_correct:
            best_correct = correct
            print('saving best model')
            torch.save(model.state_dict(), f"saved_models/dog_cat{'_base' if not args.use_slot else '_slot'}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"saved_models/dog_cat{'_base' if not args.use_slot else '_slot'}.pt")

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