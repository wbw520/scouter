#### Distributed Training
```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --world_size 4
```

#### Con-Text Dataset

##### Pre-training for MNIST dataset
```bash
python train.py --dataset MNIST --model resnet18 --batch_size 64 --epochs 10 \
--num_classes 10 --use_slot false --vis false --aug false
```

##### Positive Scouter for MNIST dataset
```bash
python train.py --dataset MNIST --model resnet18 --batch_size 64 --epochs 10 \
--num_classes 10 --use_slot true --use_pre false --loss_status 1 --slots_per_class 2 \
--power 1 --to_k_layer 1 --lambda_value 1. --vis false --channel 512 --aug false
```

##### Negative Scouter for MNIST dataset
```bash
python train.py --dataset MNIST --model resnet18 --batch_size 64 --epochs 10 \
--num_classes 10 --use_slot true --use_pre false --loss_status -1 --slots_per_class 3 \
--power 1 --to_k_layer 1 --lambda_value 1. --vis false --channel 512 --aug false
```

##### Visualization of Positive Scouter for MNIST dataset
```bash
python test.py --dataset MNIST --model resnet18 --batch_size 64 --epochs 10 \
--num_classes 10 --use_slot true --use_pre false --loss_status 1 --slots_per_class 2 \
--power 1 --to_k_layer 1 --lambda_value 1. --vis true --channel 512 --aug false
```

##### Visualization of Negative Scouter for MNIST dataset
```bash
python test.py --dataset MNIST --model resnet18 --batch_size 64 --epochs 10 \
--num_classes 10 --use_slot true --use_pre false --loss_status -1 --slots_per_class 3 \
--power 1 --to_k_layer 1 --lambda_value 1. --vis true --channel 512 --aug false
```

#### Con-Text Dataset

##### Pre-training for ConText dataset
```bash
python train.py --dataset ConText --model resnest26d --batch_size 200 --epochs 100 \
--num_classes 30 --use_slot false --vis false \
--dataset_dir /home/li/data/con-text/JPEGImages/
```

##### Positive Scouter for ConText dataset
```bash
python train.py --dataset ConText --model resnest26d --batch_size 200 --epochs 100 \
--num_classes 30 --use_slot true --use_pre true --loss_status 1 --slots_per_class 3 \
--power 2 --to_k_layer 3 --lambda_value .2 --vis false --channel 2048 \
--dataset_dir /home/li/data/con-text/JPEGImages/
```

##### Negative Scouter for ConText dataset
```bash
python train.py --dataset ConText --model resnest26d --batch_size 200 --epochs 100 \
--num_classes 30 --use_slot true --use_pre true --loss_status -1 --slots_per_class 3 \
--power 2 --to_k_layer 3 --lambda_value 1. --vis false --channel 2048 \
--dataset_dir /home/li/data/con-text/JPEGImages/
```

##### Visualization of Positive Scouter for ConText dataset
```bash
python test.py --dataset ConText --model resnest26d --batch_size 200 --epochs 100 \
--num_classes 30 --use_slot true --use_pre true --loss_status 1 --slots_per_class 3 \
--power 2 --to_k_layer 3 --lambda_value 1. --vis true --channel 2048 \
--dataset_dir /home/li/data/con-text/JPEGImages/
```

##### Visualization of Negative Scouter for ConText dataset
```bash
python test.py --dataset ConText --model resnest26d --batch_size 200 --epochs 100 \
--num_classes 30 --use_slot true --use_pre true --loss_status -1 --slots_per_class 3 \
--power 2 --to_k_layer 3 --lambda_value 1. --vis true --channel 2048 \
--dataset_dir /home/li/data/con-text/JPEGImages/
```

#### CUB-200 Dataset
##### Pre-training for CUB-200 dataset
```bash
python train.py --dataset CUB200 --model resnest50d --batch_size 64 --epochs 150 \
--num_classes 25 --use_slot false --vis false --channel 2048 \
--dataset_dir /home/wangbowen/data/bird_200/CUB_200_2011/CUB_200_2011/
```

##### Positive Scouter for CUB-200 dataset
```bash
python train.py --dataset CUB200 --model resnest50d --batch_size 64 --epochs 150 \
--num_classes 25 --use_slot true --use_pre true --loss_status 1 --slots_per_class 1 \
--power 1 --to_k_layer 3 --lambda_value 1.5 --vis false --channel 2048 \
--dataset_dir /home/wangbowen/data/bird_200/CUB_200_2011/CUB_200_2011/
```

##### Negative Scouter for CUB-200 dataset
```bash
python train.py --dataset CUB200 --model resnest50d --batch_size 64 --epochs 150 \
--num_classes 25 --use_slot true --use_pre true --loss_status -1 --slots_per_class 1 \
--power 1 --to_k_layer 3 --lambda_value 1. --vis false --channel 2048 \
--dataset_dir /home/wangbowen/data/bird_200/CUB_200_2011/CUB_200_2011/
```

##### Visualization of Positive Scouter for CUB-200 dataset
```bash
python test.py --dataset CUB200 --model resnest50d --batch_size 64 --epochs 150 \
--num_classes 25 --use_slot true --use_pre true --loss_status 1 --slots_per_class 1 \
--power 1 --to_k_layer 3 --lambda_value 1.5 --vis true --channel 2048 \
--dataset_dir /home/wangbowen/data/bird_200/CUB_200_2011/CUB_200_2011/
```

##### Visualization of Negative Scouter for CUB-200 dataset
```bash
python test.py --dataset CUB200 --model resnest50d --batch_size 64 --epochs 150 \
--num_classes 25 --use_slot true --use_pre true --loss_status -1 --slots_per_class 1 \
--power 1 --to_k_layer 3 --lambda_value 1. --vis true --channel 2048 \
--dataset_dir /home/wangbowen/data/bird_200/CUB_200_2011/CUB_200_2011/
```