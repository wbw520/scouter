#### Distributed Training
```bash
python -m torch.distributed.launch --nproc_per_node=4 --use_env train.py --world_size 4
```

#### Con-Text Dataset

##### Pre-training for ConText dataset
```bash
python train.py --dataset ConText --model resnest26d --batch_size 200 --epochs 100 \
--num_classes 30 --use_slot False --vis False \
--dataset_dir /home/li/data/con-text/JPEGImages/
```

##### Positive Scouter for ConText dataset
```bash
python train.py --dataset ConText --model resnest26d --batch_size 200 --epochs 100 \
--num_classes 30 --use_slot True --use_pre True --loss_status 1 --slots_per_class 3 \
--power 2 --to_k_layer 3 --lambda_value .2 --vis False \
--dataset_dir /home/li/data/con-text/JPEGImages/
```

##### Negative Scouter for ConText dataset
```bash
python train.py --dataset ConText --model resnest26d --batch_size 200 --epochs 100 \
--num_classes 30 --use_slot True --use_pre True --loss_status -1 --slots_per_class 3 \
--power 2 --to_k_layer 3 --lambda_value 1. --vis False \
--dataset_dir /home/li/data/con-text/JPEGImages/
```

#### CUB-200 Dataset
##### Pre-training for CUB-200 dataset
```bash
python train.py --dataset CUB200 --model resnest50d --batch_size 64 --epochs 100 \
--num_classes 25 --use_slot False --vis False \
--dataset_dir /home/wangbowen/data/bird_200/CUB_200_2011/CUB_200_2011/
```

##### Positive Scouter for CUB-200 dataset
```bash
python train.py --dataset CUB200 --model resnest50d --batch_size 64 --epochs 100 \
--num_classes 25 --use_slot True --use_pre True --loss_status 1 --slots_per_class 3 \
--power 2 --to_k_layer 3 --lambda_value 1. --vis False \
--dataset_dir /home/wangbowen/data/bird_200/CUB_200_2011/CUB_200_2011/
```

##### Negative Scouter for CUB-200 dataset
```bash
python train.py --dataset CUB200 --model resnest50d --batch_size 64 --epochs 100 \
--num_classes 25 --use_slot True --use_pre True --loss_status -1 --slots_per_class 3 \
--power 2 --to_k_layer 3 --lambda_value 1. --vis False \
--dataset_dir /home/wangbowen/data/bird_200/CUB_200_2011/CUB_200_2011/
```