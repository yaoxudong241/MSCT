import torch.utils.data as data
import os.path
import cv2
import numpy as np
from dataset import common
import argparse, os
from torch.utils.data import DataLoader
import rasterio
import torch
from pathlib import Path
import random
def default_loader(path):
    with rasterio.open(path) as src:
        meta = src.meta
        nrows, ncols = src.shape
        data =src.read()
        data = np.transpose(data, (1, 2, 0))
    return data

def npy_loader(path):
    return np.load(path,allow_pickle=True)

IMG_EXTENSIONS = [
    '.tif',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


class tifdataset(data.Dataset):
    def __init__(self, opt,val):
        self.opt = opt
        self.val = val

        if (not self.val):
            self.root = self.opt.root
        else:
            self.root = self.opt.rootval

        self.ext = self.opt.ext
        self.train = True if self.opt.phase == 'train' else False
        self.repeat = 1
        # self.opt.test_every // (self.opt.n_train // self.opt.batch_size)
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr = self._scan()

    def _set_filesystem(self, dir_data):
        self.root = dir_data   # 改
        self.dir_hr = os.path.join(self.root, 'HR')  # 改
        self.dir_lr = os.path.join(self.root, 'LR\X2')   # 改


    def __getitem__(self, idx):
        lr, hr, lr_name, hr_name = self._load_file(idx)
        lr, hr = self._get_patch(lr, hr)
        lr_tensor = torch.from_numpy(lr.copy().astype(np.float32)).permute(2, 0, 1)/10000
        hr_tensor = torch.from_numpy(hr.copy().astype(np.float32)).permute(2, 0, 1)/10000
        return lr_tensor, hr_tensor, lr_name, hr_name

    def __len__(self):
        if self.val:
            return self.opt.n_val * self.repeat
        if self.train:
            return self.opt.n_train * self.repeat


    def _get_index(self, idx):
        if self.val:
            return idx % self.opt.n_val
        if self.train:
            return idx % self.opt.n_train


    def _get_patch(self, img_in, img_tar):

        if (not self.val):
            patch_size = self.opt.patch_size
        else:
            patch_size = 128




        img_in, img_tar = common.get_patch(
            img_in, img_tar, patch_size=patch_size)
        img_in, img_tar = common.augment(img_in, img_tar)
        return img_in, img_tar

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr = default_loader(self.images_lr[idx])
        hr = default_loader(self.images_hr[idx])
        return lr, hr, self.images_lr[idx], self.images_hr[idx]


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16,
                        help="training batch size")

    parser.add_argument("--threads", type=int, default=0,
                        help="number of threads for data loading")
    parser.add_argument("--root", type=str, default=r"C:\Users\BUAA_D723_4\Documents\Landsat_and_MODIS_data_for_the_Coleambally_Irrigation_Area-XlqmtwZ5-\data\CIA\train",  # 改
                        help='dataset directory')
    parser.add_argument("--n_train", type=int, default=787,  # 改
                        help="number of training set")
    parser.add_argument("--patch_size", type=int, default=96,
                        help="output patch size")
    parser.add_argument("--isY", action="store_true", default=True)
    parser.add_argument("--ext", type=str, default='.tif')  # 改
    parser.add_argument("--phase", type=str, default='train')
    parser.add_argument("--rootval", type=str,
                        default=r"C:\Users\BUAA_D723_4\Documents\Landsat_and_MODIS_data_for_the_Coleambally_Irrigation_Area-XlqmtwZ5-\data\CIA\test",
                        # 改
                        help='dataset directory')
    args = parser.parse_args()

    trainset = tifdataset(args,True)

    training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=1,
                                      shuffle=False)

    for iteration, (lr_tensor, hr_tensor, reflr_tensor, refhr_tensor) in enumerate(training_data_loader, 1):
        print(lr_tensor.size())





