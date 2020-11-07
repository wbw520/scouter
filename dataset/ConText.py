from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tools.prepare_things import get_name
import os
import torch
import numpy as np


class MakeList(object):
    """
    this class used to make list of data for model train and test, return the root name of each image
    root: txt file records condition for every cxr image
    """
    def __init__(self, args, ratio=0.8):
        self.image_root = args.dataset_dir
        self.all_image = get_name(self.image_root, mode_folder=False)
        self.category = sorted(set([i[:i.find('_')] for i in self.all_image]))

        for c_id, c in enumerate(self.category):
            print(c_id, '\t', c)

        self.ration = ratio

    def get_data(self):
        all_data = []
        for img in self.all_image:
            label = self.deal_label(img)
            all_data.append([os.path.join(self.image_root, img), label])
        train, val = train_test_split(all_data, random_state=1, train_size=self.ration)
        return train, val

    def deal_label(self, img_name):
        categoty_no = img_name[:img_name.find('_')]
        back = self.category.index(categoty_no)
        return back


class MakeListImage():
    """
    this class used to make list of data for ImageNet
    """
    def __init__(self, args):
        self.image_root = args.dataset_dir
        self.category = get_name(self.image_root + "train/")
        self.used_cat = self.category[:args.num_classes]
        # for c_id, c in enumerate(self.used_cat):
        #     print(c_id, '\t', c)

    def get_data(self):
        train = self.get_img(self.used_cat, "train")
        val = self.get_img(self.used_cat, "val")
        return train, val

    def get_img(self, folders, phase):
        record = []
        for folder in folders:
            current_root = os.path.join(self.image_root, phase, folder)
            images = get_name(current_root, mode_folder=False)
            for img in images:
                record.append([os.path.join(current_root, img), self.deal_label(folder)])
        return record

    def deal_label(self, img_name):
        back = self.used_cat.index(img_name)
        return back


class ConText(Dataset):
    """read all image name and label"""
    def __init__(self, data, transform=None):
        self.all_item = data
        self.transform = transform

    def __len__(self):
        return len(self.all_item)

    def __getitem__(self, item_id):  # generate data when giving index
        while not os.path.exists(self.all_item[item_id][0]):
            raise ("not exist image:" + self.all_item[item_id][0])
        image_path = self.all_item[item_id][0]
        image = Image.open(image_path).convert('RGB')
        if image.mode == 'L':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.all_item[item_id][1]
        label = torch.from_numpy(np.array(label))
        return {"image": image, "label": label, "names": image_path}