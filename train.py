import argparse
from pathlib import Path
import torch
from torch.utils.data import DistributedSampler
import tools.prepare_things as prt
from engine import train_one_epoch, evaluate
import tools.data_loader as bird
from tools.data_loader import DataLoaderX
from sloter.slot_model import SlotModel
import datetime
import time


def get_args_parser():
    parser = argparse.ArgumentParser('Set 3D model', add_help=False)
    parser.add_argument('--model', default="resnest50d", type=str)

    # training set
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--lr_drop', default=70, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument("--num_classes", default=50, type=int)
    parser.add_argument('--img_size', default=260, help='path for save data')
    parser.add_argument('--pre_trained', default=True, help='whether use pre parameter for backbone')
    parser.add_argument('--use_slot', default=True, help='whether use slot module')
    parser.add_argument('--use_pre', default=True, help='whether use pre dataset parameter')

    # slot setting
    parser.add_argument('--loss_status', default=-1, help='positive or negetive loss')
    parser.add_argument('--hidden_dim', default=64, help='dimension of to_k')
    parser.add_argument('--slots_per_class', default=1, help='number of slot for each class')
    parser.add_argument('--vis', default=False, help='whether save slot visualization')
    parser.add_argument('--vis_id', default=0, help='choose image to visualization')

    # data/machine set
    parser.add_argument('--dataset_dir', default='/home/wangbowen/data/bird_200/CUB_200_2011/CUB_200_2011/',
                        help='path for save data')
    parser.add_argument('--output_dir', default='saved_model/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--pre_dir', default='pre_model/',
                        help='path of pre-train model')
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--resume', default=False, help='resume from checkpoint')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def main(args):
    prt.init_distributed_mode(args)
    device = torch.device(args.device)

    model = SlotModel(args)
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    dataset_train = bird.CUB_200(args, train=True, transform=bird.make_video_transform("train"))
    dataset_val = bird.CUB_200(args, train=False, transform=bird.make_video_transform("val"))
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoaderX(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers, pin_memory=True)
    data_loader_val = DataLoaderX(dataset_val, args.batch_size, sampler=sampler_val, num_workers=args.num_workers, pin_memory=True)
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
    record = {"train": {"loss": [], "acc_1": [], "acc_5": []},
              "val": {"loss": [], "acc_1": [], "acc_5": []}}
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_one_epoch(model, optimizer, data_loader_train, device, criterion, record, epoch)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / (f"{'use_slot_' if args.use_slot else 'no_slot_'}" + f"{'negetive_' if args.use_slot and args.loss_status != 1 else ''}" + 'checkpoint.pth')]
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 10 == 0:
                checkpoint_paths.append(output_dir / (f"{'use_slot_' if args.use_slot else 'no_slot_'}" + f"{'negetive_' if args.use_slot and args.loss_status != 1 else ''}" + f'checkpoint{epoch:04}.pth'))
            for checkpoint_path in checkpoint_paths:
                prt.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        evaluate(model, data_loader_val, device, criterion, record, epoch)

        print("train loss:", record["train"]["loss"])
        print("val loss:", record["val"]["loss"])
        print("train acc_1:", record["train"]["acc_1"])
        print("val acc_1:", record["val"]["acc_1"])
        print("train acc_5:", record["train"]["acc_5"])
        print("val acc_5:", record["val"]["acc_5"])

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)