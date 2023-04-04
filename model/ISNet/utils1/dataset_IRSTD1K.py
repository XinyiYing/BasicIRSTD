import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
import random
import cv2
from albumentations import (
    RandomRotate90, Transpose, ShiftScaleRotate, Blur,
    OpticalDistortion, CLAHE, GaussNoise, MotionBlur,
    GridDistortion, HueSaturationValue,ToGray,
    MedianBlur, PiecewiseAffine, Sharpen, Emboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
import numpy as np

def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        ToGray(),
        OneOf([
            GaussNoise(),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            PiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            Sharpen(),
            Emboss(),
            RandomBrightnessContrast()
        ], p=0.5),
        HueSaturationValue(p=0.5),
    ], p=p)










class SirstDataset(Data.Dataset):

    def __init__(self, args, mode='train'):

        base_dir = '/home/imglab/ZR/ISNet_/IRSTD-1k'

        if mode == 'train':
            txtfile = 'trainval.txt'

        elif mode == 'val':
            txtfile = 'test.txt'

        self.list_dir = osp.join(base_dir, txtfile)
        self.imgs_dir = osp.join(base_dir, 'IRSTD1k_Img')
        self.label_dir = osp.join(base_dir, 'IRSTD1k_Label')
        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.mode = mode
        self.crop_size = args.crop_size
        self.base_size = args.base_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.imgs_dir, name+'.png')
        label_path = osp.join(self.label_dir, name+'.png')


        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)

        if self.mode == 'train':

            # img, mask = self._sync_transform(img, mask)
            img, mask = self._sync_transform(img, mask)
            edgemap = mask
        elif self.mode == 'val':
            img, mask = self._testval_sync_transform(img, mask)
            edgemap = mask
        else:
            raise ValueError("Unkown self.mode")

        #img = self.transform(img)
        # print(max(mask))
        # print(min(mask))

        img, mask, edgemap = self.transform(img), transforms.ToTensor()(mask), transforms.ToTensor()(edgemap)
        # if mask.size()[0] == 4:
        #     mask = mask[0:3,:,:]
        # if img.size()[0] == 4:
        #     img = img[0:3, :, :]
        # if edgemap.size()[0] == 4:
        #     edgemap = edgemap[0:3, :, :]
        # print(img.size())
        # print('mask', mask.size()[0])
        return img, mask, edgemap

    def __len__(self):
        return len(self.names)

    def __filename__(self):
        return self.names

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
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        #Albu aug
        img_1 = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        mask_1 = cv2.cvtColor(np.asarray(mask),cv2.COLOR_RGB2BGR)
        data = {"image": img_1, "mask": mask_1}
        augmentation = strong_aug(p=1.0)
        augmented = augmentation(**data)  ## 数据增强
        img_1, mask_1 = augmented["image"], augmented["mask"]
        img = Image.fromarray(cv2.cvtColor(img_1,cv2.COLOR_BGR2RGB))
        mask = Image.fromarray(cv2.cvtColor(mask_1, cv2.COLOR_BGR2RGB))

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
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))

        return img, mask

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask
