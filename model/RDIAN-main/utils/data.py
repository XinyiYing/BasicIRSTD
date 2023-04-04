import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
import random

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def _iter_(self):
        return BackgroundGenerator(super()._iter_())

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=3, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

class IRDSTDataset(Data.Dataset):
    def __init__(self, args, mode='train'):   
        #path of dataset
        base_dir = 'datasets/example/'

        # list of images in dataset
        if mode == 'train':
            txtfile = 'train.txt'
        elif mode == 'val':
            # txtfile = 'val.txt'
            txtfile = 'test.txt'

        self.list_dir = osp.join(base_dir,  txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.mask_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.crop_size = args.crop_size
        self.base_size = args.base_size
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize([-0.1246], [1.0923]) # mean and std
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = self.imgs_dir+'/'+name+'.bmp'
        mask_path = self.mask_dir+'/'+name+'.bmp'

        img = Image.open(img_path).convert('L')  
        mask = Image.open(mask_path).convert('1')

        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
        else:
            raise ValueError("Unkown self.mode")

        img, mask = self.transform(img), transforms.ToTensor()(mask)
        return img, mask

    def __len__(self):
        return len(self.names)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT) 
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        crop_size = self.crop_size
        # random scale (short edge)
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        return img, mask

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # # center crop
        # w, h = img.size
        # x1 = int(round((w - outsize) / 2.))
        # y1 = int(round((h - outsize) / 2.))
        # img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        # mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))


        return img, mask

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask

