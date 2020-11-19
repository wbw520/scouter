import numpy as np
import torch
from sloter.slot_model import load_backbone
import argparse
from train import get_args_parser
from torchvision import datasets, transforms
from matplotlib.colors import LinearSegmentedColormap
import os
from PIL import Image
import torch.nn.functional as F
from sloter.utils.vis import apply_colormap_on_image
from torchcam.cams import CAM, GradCAM, GradCAMpp, SmoothGradCAMpp, ScoreCAM, SSCAM, ISSCAM
from torchcam.utils import overlay_mask

from tqdm import tqdm
from dataset.ConText import ConText, MakeList, MakeListImage
from dataset.CUB200 import CUB_200

from torchcam.IGOS import Get_blurred_img, Integrated_Mask
from torchray.attribution.rise import rise
from torchray.attribution.extremal_perturbation import extremal_perturbation
from torchcam.IBA.pytorch import IBA, tensor_to_np_img, get_imagenet_folder, imagenet_transform
from torchcam.IBA.utils import plot_saliency_map, to_unit_interval, load_monkeys
from torch.utils.data import DataLoader

def show_cam_on_image(img, masks, target_index, save_name):
    final = np.uint8(255*masks)

    mask_image = Image.fromarray(final, mode='L')
    mask_image.save(f'sloter/vis/{save_name}_{target_index}_mask.png')

    heatmap_only, heatmap_on_image = apply_colormap_on_image(img, final, 'jet')
    heatmap_on_image.save(f'sloter/vis/{save_name}_{target_index}.png')


def make_grad(extractor, output, img_heat, grad_min_level, save_name):
    img_heat = img_heat.resize((args.img_size, args.img_size), Image.BILINEAR)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    # target_index = None
    for target_index in tqdm(range(0, args.num_classes)):
        mask = extractor(target_index, output).cpu().unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=(args.img_size, args.img_size), mode="bilinear")
        mask = mask.squeeze(dim=0).squeeze(dim=0)
        mask = mask.detach().numpy()
        mask = np.maximum(mask, 0)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        mask = np.maximum(mask, grad_min_level)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        show_cam_on_image(img_heat, mask, target_index, save_name)


def for_vis(args):
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
        image = data["image"][0]
        label = data["label"][0]
        image_orl = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8).transpose((1,2,0)), mode='RGB')
        image = transform(image_orl)
        transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    elif args.dataset == 'ImageNet':
        train, val = MakeListImage(args).get_data()
        dataset_val = ConText(val, transform=transform)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
        iter_loader = iter(data_loader_val)
        for i in range(0, 1):
            data = iter_loader.next()
        image = data["image"][0]
        label = data["label"][0].item()
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
        dataset_val = CUB_200(args, train=False, transform=transform)
        data_loader_val = torch.utils.data.DataLoader(dataset_val, args.batch_size, shuffle=False, num_workers=1, pin_memory=True)
        data = iter(data_loader_val).next()
        image = data["image"][0]
        label = data["label"][0]
        image_orl = Image.fromarray((image.cpu().detach().numpy()*255).astype(np.uint8).transpose((1,2,0)), mode='RGB')
        image = transform(image_orl)
        transform = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = transform(image)
    image = image.unsqueeze(0)
    device = torch.device(args.device)


    ### IGOS
    model = load_backbone(args)
    model = model.to(device)
    model.eval()

    image_orl_for_blur = np.float32(image_orl) / 255.
    img, blurred_img, logitori = Get_blurred_img(image_orl_for_blur, label, model, resize_shape=(260, 260),
                                                    Gaussian_param=[51, 50],
                                                    Median_param=11, blur_type='Gaussian', use_cuda=1)

    for target_index in tqdm(range(0, args.num_classes)):
        mask, upsampled_mask, imgratio, curvetop, curve1, curve2, category = Integrated_Mask(img, blurred_img, model,
                                                                                                    label,
                                                                                                    max_iterations=15,
                                                                                                    integ_iter=20,
                                                                                                    tv_beta=2,
                                                                                                    l1_coeff=0.01 * 100,
                                                                                                    tv_coeff=0.2 * 100,
                                                                                                    size_init=8,
                                                                                                    use_cuda=1)  #
        mask = upsampled_mask.cpu().detach().numpy()[0,0]
        mask = -mask + mask.max()*2.
        
        mask = np.maximum(mask, 0)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        mask = np.maximum(mask, args.grad_min_level)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        # mask = Image.fromarray(mask*255, mode='L').resize((args.img_size, args.img_size), Image.BILINEAR)
        # mask = np.uint8(mask)

        image_orl = image_orl.resize((args.img_size, args.img_size), Image.BILINEAR)
        # heatmap = np.array(heatmap)
        show_cam_on_image(image_orl, mask, target_index, 'IGOS')

    del model


    ### torchray (RISE)
    model = load_backbone(args)
    model = model.to(device)
    model.eval()

    for target_index in tqdm(range(0, args.num_classes)):
        mask = rise(model, image.to(device), target_index)
        mask = mask.cpu().numpy()[0,0]

        mask = np.maximum(mask, 0)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        mask = np.maximum(mask, args.grad_min_level)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)

        image_orl = image_orl.resize((args.img_size, args.img_size), Image.BILINEAR)
        # heatmap = np.array(heatmap)
        show_cam_on_image(image_orl, mask, target_index, 'RISE')


    del model


    ### torchray (Extremal)
    model = load_backbone(args)
    model = model.to(device)
    model.eval()

    for target_index in tqdm(range(0, args.num_classes)):
        mask, _ = extremal_perturbation(model, image.to(device), target_index)
        mask = mask.cpu().numpy()[0,0]

        mask = np.maximum(mask, 0)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        mask = np.maximum(mask, args.grad_min_level)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)

        image_orl = image_orl.resize((args.img_size, args.img_size), Image.BILINEAR)
        # heatmap = np.array(heatmap)
        show_cam_on_image(image_orl, mask, target_index, 'Extremal')


    del model

    ### IBA
    model = load_backbone(args)
    model = model.to(device)
    model.eval()

    imagenet_dir = '../../data/imagenet/ILSVRC/Data/CLS-LOC/validation'
    # Add a Per-Sample Bottleneck at layer conv4_1
    iba = IBA(model.layer4)

    # Estimate the mean and variance of the feature map at this layer.
    val_set = get_imagenet_folder(imagenet_dir)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True, num_workers=4)
    iba.estimate(model, val_loader, n_samples=5000, progbar=True)

    for target_index in tqdm(range(0, args.num_classes)):
        # Closure that returns the loss for one batch
        model_loss_closure = lambda x: -torch.log_softmax(model(x.to(device)), dim=1)[:, target_index].mean()
        # Explain class target for the given image
        saliency_map = iba.analyze(image, model_loss_closure, beta=10)
        # display result
        model_loss_closure = lambda x: -torch.log_softmax(model(x.to(device)), 1)[:, target_index].mean()
        heatmap = iba.analyze(image, model_loss_closure )

        mask = heatmap
        mask = np.maximum(mask, 0)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        mask = np.maximum(mask, args.grad_min_level)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)

        image_orl = image_orl.resize((args.img_size, args.img_size), Image.BILINEAR)
        # heatmap = np.array(heatmap)
        show_cam_on_image(image_orl, mask, target_index, 'IBA')
        # plot_saliency_map(heatmap, tensor_to_np_img(image[0]))

    RESNET_CONFIG = dict(input_layer='conv1', conv_layer='layer4', fc_layer='fc')

    MODEL_CONFIG = {**RESNET_CONFIG}
    conv_layer = MODEL_CONFIG['conv_layer']
    input_layer = MODEL_CONFIG['input_layer']
    fc_layer = MODEL_CONFIG['fc_layer']

    ### torchcam
    del model
    model = load_backbone(args)
    model = model.to(device)
    model.eval()
    # Hook the corresponding layer in the model
    cam_extractors = [CAM(model, conv_layer, fc_layer), GradCAM(model, conv_layer),
                      GradCAMpp(model, conv_layer), SmoothGradCAMpp(model, conv_layer, input_layer),
                      ScoreCAM(model, conv_layer, input_layer),
                      SSCAM(model, conv_layer, input_layer),
                    #   ISSCAM(model, conv_layer, input_layer), 
                      ]
    cam_extractors_names = ['CAM', 'GradCAM',
                      'GradCAMpp', 'SmoothGradCAMpp',
                      'ScoreCAM',
                      'SSCAM',
                    #   'ISSCAM', 
                      ]                
    for idx, extractor in enumerate(cam_extractors):
        model.zero_grad()

        output1 = model(image.to(device))
        output = F.softmax(output1, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)

        pred_label_idx.squeeze_()
        predicted_label = str(pred_label_idx.item())
        print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

        make_grad(extractor, output1, image_orl, args.grad_min_level, cam_extractors_names[idx])
        extractor.clear_hooks()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    args_dict = vars(args)
    args_for_evaluation = ['num_classes', 'lambda_value', 'power', 'slots_per_class']
    args_type = [int, float, int, int]
    for arg_id, arg in enumerate(args_for_evaluation):
        args_dict[arg] = args_type[arg_id](args_dict[arg])

    os.makedirs('sloter/vis', exist_ok=True)
    for_vis(args)
