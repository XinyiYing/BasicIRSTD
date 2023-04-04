import torch
import torch.nn as nn
from torchvision.transforms import functional as F
from torchvision import transforms as T

import cv2
import numpy as np

from model import Network

class detector:
    def __init__(self, oup='ds', int8=False):
        self.model = Network()

        modelWeightPath = './save_pth/ISTDU_Net/best.pth'

        self.device = 'cuda'
        self.model = self.model.to(self.device)
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(modelWeightPath))
        self.model.eval()
        # self.normTransform = T.Normalize(0.5, 0.25)
        self.normTransform = T.Normalize(0, 1)
        self.oup = oup
        self.int8 = int8

    def __call__(self, img):
        # h, w = img.shape
        img = self.pocessing(img)
        if self.oup == 'ds':
            _, output = self.model(img)
            # output, _ = self.model(img)
            output = output.detach().cpu().numpy()
            resultMask = output[0][0]
            if self.int8:
                resultMask = np.uint8(resultMask*255)
        elif self.oup == 'dsadd':
            a, b = self.model(img)
            a = a.detach().cpu().numpy()[0][0]
            b = b.detach().cpu().numpy()[0][0]
            x = np.array([a, b])
            output = np.max(x, axis=0)
            resultMask = output
            if self.int8:
                resultMask = np.uint8(resultMask*255)
        else:
            output = self.model(img)
            output = output.detach().cpu().numpy()
            resultMask = output[0][0]
            if self.int8:
                resultMask = np.uint8(resultMask*255)
        # resultMask = cv2.resize(resultMask, (w, h))
        # resultMask[resultMask > 0] = 255
        return resultMask

    def pocessing(self, img):
        # img = cv2.resize(img, (512, 512))
        # img = cv2.resize(img, (800, 608))

        # img = torch.tensor(img).to(self.device).unsqueeze(0).unsqueeze(0)
        # img = img / 255
        # img = self.normTransform(img)
        # return img

        img = F.to_pil_image(img)
        img = F.to_tensor(img)
        img = self.normTransform(img).unsqueeze(0)
        return img


if __name__ == '__main__':
    d = detector()
    path = './test/images/Misc_2.png' ##change to the img path where you want to test the image
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (512, 512))
    out = d(img)
    cv2.imshow('img', img)
    cv2.imshow('out', out)
    cv2.waitKey(0)