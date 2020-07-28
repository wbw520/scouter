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
from utils.mnist_model import Mnist_Model, Mnist_Resnet_Model

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


        output, attn_loss = model(data, selected_classes=selected_classes)
        loss = F.nll_loss(output, target) + attn_loss
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
            output, _ = model(data)
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
    parser.add_argument('--slots', type=int, default=10, metavar='N',
                        help=' (default: 10)')
    parser.add_argument('--slots_per_class', type=int, default=1, metavar='N',
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
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                     )

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Mnist_Resnet_Model(slots_num=args.slots, slots_per_class=args.slots_per_class, slot=args.use_slot, pre_trained=args.pre_trained).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    best_correct = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, args.slots)
        correct = test(model, device, test_loader)
        scheduler.step()
        if correct >= best_correct:
            best_correct = correct
            print('saving best model')
            torch.save(model.state_dict(), f"saved_models/mnist_resnet{'_base' if not args.use_slot else '_slot'}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"saved_models/mnist_resnet{'_base' if not args.use_slot else '_slot'}.pt")


if __name__ == '__main__':
    main()