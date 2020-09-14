import argparse
from pathlib import Path
import torch
from torch.utils.data import DistributedSampler
import tools.prepare_things as prt
from engine import train_one_epoch, evaluate
from dataset.choose_dataset import select_dataset
from tools.prepare_things import DataLoaderX
from sloter.slot_model import SlotModel
from tools.calculate_tool import MetricLog
import datetime
import time
import numpy as np
from thop import profile, clever_format
import tensorly as tl


def get_args_parser():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser('Set SCOUTER model', add_help=False)
    parser.add_argument('--model', default="resnet18", type=str)
    parser.add_argument('--dataset', default="MNIST", type=str)
    parser.add_argument('--channel', default=512, type=int)

    # training set
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_drop', default=70, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument("--num_classes", default="10", type=str)
    parser.add_argument('--img_size', default=260, help='path for save data')
    parser.add_argument('--pre_trained', default=True, type=str2bool, help='whether use pre parameter for backbone')
    parser.add_argument('--use_slot', default=True, type=str2bool, help='whether use slot module')
    parser.add_argument('--use_pre', default=False, type=str2bool, help='whether use pre dataset parameter')
    parser.add_argument('--aug', default=False, type=str2bool, help='whether use pre dataset parameter')
    parser.add_argument('--grad', default=False, type=str2bool, help='whether use grad-cam for visulazition')
    parser.add_argument('--grad_min_level', default=0., type=float, help='control the grad-cam vis area')
    parser.add_argument('--iterated_evaluation_num', default=1, type=int, help='used for iterated evaluation')
    parser.add_argument('--cal_area_size', default=False, type=str2bool, help='whether to calculate for area size of the attention map')
    parser.add_argument('--thop', default=False, type=str2bool, help='whether to only calculate for the model costs (no training)')

    # slot setting
    parser.add_argument('--loss_status', default=1, type=int, help='positive or negative loss')
    parser.add_argument('--freeze_layers', default=2, type=int, help='number of freeze layers')
    parser.add_argument('--hidden_dim', default=64, type=int, help='dimension of to_k')
    parser.add_argument('--slots_per_class', default="3", type=str, help='number of slot for each class')
    parser.add_argument('--power', default="2", type=str, help='power of the slot loss')
    parser.add_argument('--to_k_layer', default=1, type=int, help='number of layers in to_k')
    parser.add_argument('--lambda_value', default="1.", type=str, help='lambda of slot loss')
    parser.add_argument('--vis', default=False, type=str2bool, help='whether save slot visualization')
    parser.add_argument('--vis_id', default=0, type=int, help='choose image to visualization')

    # data/machine set
    parser.add_argument('--dataset_dir', default='../PAN/bird_200/CUB_200_2011/CUB_200_2011/',
                        help='path for save data')
    parser.add_argument('--output_dir', default='saved_model/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--pre_dir', default='pre_model/',
                        help='path of pre-train model')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--resume', default=False, type=str2bool, help='resume from checkpoint')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    prt.init_distributed_mode(args)
    device = torch.device(args.device)

    model = SlotModel(args)
    print("train model: " + f"{'use slot ' if args.use_slot else 'without slot '}" + f"{'negetive loss' if args.use_slot and args.loss_status != 1 else 'positive loss'}")
    model.to(device)
    model_without_ddp = model

    if args.thop:
        def freeze_layers(model):
            for layer in model.children():
                if isinstance(layer, torch.nn.Sequential):
                    for sub_layer in layer:
                        sub_layer.requires_grad = False
                        for parameter in sub_layer.parameters():
                            parameter.requires_grad = False
                else:
                    layer.requires_grad = False
                    for parameter in layer.parameters():
                        parameter.requires_grad = False

        def unfreeze_layers(model):
            for layer in model.children():
                if isinstance(layer, torch.nn.Sequential):
                    for sub_layer in layer:
                        sub_layer.requires_grad = True
                        for parameter in sub_layer.parameters():
                            parameter.requires_grad = True
                else:
                    layer.requires_grad = True
                    for parameter in layer.parameters():
                        parameter.requires_grad = True

        unfreeze_layers(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(float(n_parameters)/1000000, 'M')

        
        freeze_layers(model)
        model.cpu()
        model.eval()
        tl.set_backend('pytorch')

        input_ = torch.randn(1, 3, 260, 260)

        flops_list = []
        params_list = []
        acc_list = []

        flops, params = profile(model, inputs=(input_, ))
        flops_list.append(flops)
        params_list.append(params)
        flops, params = clever_format([flops, params], "%.3f")
        print(float(n_parameters)/1000000, 'M', params, flops)
        return [float(n_parameters)/1000000, flops_list[-1]/1000000000]

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    dataset_train, dataset_val = select_dataset(args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoaderX(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    data_loader_val = DataLoaderX(dataset_val, args.batch_size, sampler=sampler_val, num_workers=args.num_workers)
    output_dir = Path(args.output_dir)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    print("Start training")
    start_time = time.time()
    log = MetricLog()
    record = log.record
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_one_epoch(model, data_loader_train, optimizer, device, record, epoch)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / (f"{args.dataset}_" + f"{'use_slot_' if args.use_slot else 'no_slot_'}"\
                 + f"{'negative_' if args.use_slot and args.loss_status != 1 else ''}"\
                      + f"{'for_area_size_'+str(args.lambda_value) + '_'+ str(args.slots_per_class) + '_' if args.cal_area_size else ''}" + 'checkpoint.pth')]
            # extra checkpoint before LR drop and every 10 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / (f"{args.dataset}_" + f"{'use_slot_' if args.use_slot else 'no_slot_'}"\
                     + f"{'negative_' if args.use_slot and args.loss_status != 1 else ''}"\
                          + f"{'for_area_size_'+str(args.lambda_value) + '_'+ str(args.slots_per_class) + '_' if args.cal_area_size else ''}" + f'checkpoint{epoch:04}.pth'))
            for checkpoint_path in checkpoint_paths:
                prt.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        evaluate(model, data_loader_val, device, record, epoch)
        log.print_metric()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    return [record["train"]["acc"][-1], record["val"]["acc"][-1]]


def param_translation(args):
    args_dict = vars(args)
    args_for_evaluation = ['num_classes', 'lambda_value', 'power', 'slots_per_class']
    args_type = [int, float, int, int]
    target_arg = None
    for arg_id, arg in enumerate(args_for_evaluation):
        if args_dict[arg].find(',') > 0:
            target_arg = arg
            target_type = args_type[arg_id]
            setting_list = args_dict[arg].split(",")
        else:
            args_dict[arg] = args_type[arg_id](args_dict[arg])

    if target_arg is None:
        main(args)
    else:
        record = {}
        circle_turns = args.iterated_evaluation_num
        for set in setting_list:
            record.update({f"{target_arg}-"+set: []})
            args_dict[target_arg] = target_type(set)
            for turn in range(circle_turns):
                record[f"{target_arg}-"+set].append(main(args))
                print(record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    param_translation(args)
