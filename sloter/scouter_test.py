from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import numpy as np
from utils.mnist_model import Mnist_Model, Mnist_Resnet_Model#, CamExtractor
from utils.grad_cam import GradCam, save_class_activation_images
from utils.vis import apply_colormap_on_image


def test(model, device, test_loader):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        model.eval()
        data, target = data.to(device), target.to(device)
        output = torch.sigmoid(model(data, selected_classes=None)[0])
        # for i in range(9,1,-4):
        #     tok3 = torch.topk(output, i, dim=1).indices
        #     selected_classes = np.zeros((data.size(0), 10))
        #     for b_id, b in enumerate(tok3):
        #         for t in b:
        #             selected_classes[b_id, t] = 1
        #     # selected_classes[:,7] = 1
        #     # selected_classes[:,1] = 1
        #     # selected_classes = None
        #     # selected_classes = selected_classes.cuda()
        #     output = torch.sigmoid(model(data, selected_classes=selected_classes))
        #     if selected_classes is not None:
        #         selected_classes = torch.from_numpy(selected_classes).float().cuda()
        #         output = torch.einsum('bi,bi->bi', output, selected_classes)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        #For vis
        image_raw = Image.fromarray((data.cpu().detach().numpy()*255).astype(np.uint8)[0,0], mode='L')#.resize((400,400))
        image_raw.save('vis/image.png')
        print(torch.argmax(output[0]).item())
        model.train()
        # grad_cam = GradCam(model, target_layer='conv2', cam_extractor=CamExtractor)
        trans = transforms.ToTensor()

        for id in range(10):
            image_raw = Image.open('vis/image.png').convert('RGB')
            # image_raw_cam = Image.open('vis/image.png')
            slot_image = np.array(Image.open(f'vis/slot_{id}.png').resize(image_raw.size, resample=Image.BILINEAR), dtype=np.uint8)

            heatmap_only, heatmap_on_image = apply_colormap_on_image(image_raw, slot_image, 'gist_rainbow_r')
            heatmap_on_image.save(f'vis/slot_mask_{id}.png')
        
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
    parser.add_argument('--slots', type=int, default=10, metavar='N',
                        help=' (default: 10)')
    parser.add_argument('--slots_per_class', type=int, default=1, metavar='N',
                        help=' (default: 1)')
    parser.add_argument('--use_slot', type=bool, default=True, metavar='N',
                        help=' (default: True)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False},
                     )

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Mnist_Resnet_Model(vis=True, slots_num=args.slots, slots_per_class=args.slots_per_class).to(device)
    model.load_state_dict(torch.load(f"saved_models/mnist_resnet{'_base' if not args.use_slot else '_slot'}.pt"))

    test(model, device, test_loader)


if __name__ == '__main__':
    main()