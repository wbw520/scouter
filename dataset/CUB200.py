from PIL import Image
from torch.utils.data import Dataset
import os
import torch
import numpy as np


class CUB_200(Dataset):
    def __init__(self, args, train=True, transform=None):
        super(CUB_200, self).__init__()
        self.root = args.dataset_dir
        self.size = args.img_size
        self.num = args.num_classes
        self.train = train
        self.transform_ = transform
        self.classes_file = os.path.join(self.root, 'classes.txt')  # <class_id> <class_name>
        self.image_class_labels_file = os.path.join(self.root, 'image_class_labels.txt')  # <image_id> <class_id>
        self.images_file = os.path.join(self.root, 'images.txt')  # <image_id> <image_name>
        self.train_test_split_file = os.path.join(self.root, 'train_test_split.txt')  # <image_id> <is_training_image>
        self.bounding_boxes_file = os.path.join(self.root, 'bounding_boxes.txt')  # <image_id> <x> <y> <width> <height>

        self._train_ids = []
        self._test_ids = []
        self._image_id_label = {}
        self._train_path_label = []
        self._test_path_label = []

        self._train_test_split()
        self._get_id_to_label()
        self._get_path_label()

    def _train_test_split(self):

        for line in open(self.train_test_split_file):
            image_id, label = line.strip('\n').split()
            if label == '1':
                self._train_ids.append(image_id)
            elif label == '0':
                self._test_ids.append(image_id)
            else:
                raise Exception('label error')

    def _get_id_to_label(self):
        for line in open(self.image_class_labels_file):
            image_id, class_id = line.strip('\n').split()
            self._image_id_label[image_id] = class_id

    def _get_path_label(self):
        for line in open(self.images_file):
            image_id, image_name = line.strip('\n').split()
            if int(image_name[:3]) > self.num:
                if image_id in self._train_ids:
                    self._train_ids.pop(self._train_ids.index(image_id))
                else:
                    self._test_ids.pop(self._test_ids.index(image_id))
                continue
            label = self._image_id_label[image_id]
            if image_id in self._train_ids:
                self._train_path_label.append((image_name, label))
            else:
                self._test_path_label.append((image_name, label))

    def __getitem__(self, index):
        if self.train:
            image_name, label = self._train_path_label[index]
        else:
            image_name, label = self._test_path_label[index]
        image_path = os.path.join(self.root, 'images', image_name)
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        label = int(label) - 1
        label = torch.from_numpy(np.array(label))
        if self.transform_ is not None:
            img = self.transform_(img)
        return {"image": img, "label": label, "names": image_path}

    def __len__(self):
        if self.train:
            return len(self._train_ids)
        else:
            return len(self._test_ids)
