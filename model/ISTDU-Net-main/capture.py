import cv2
import numpy as np
import os
import glob
import scipy.io as scio

class matCapture:
    def __init__(self, path, gray=False):
        self.path = path
        # self.images = sorted(glob.glob(os.path.join(self.path, '*.*g')))
        # self.images = sorted(glob.glob(os.path.join(self.path, '*.bmp')), key=lambda x: int(os.path.basename(x).split('.')[0]))

        self.images = glob.glob(os.path.join(self.path, '*.*'))
        self.images = list(filter(lambda x: os.path.basename(x).split('.')[1] in ['mat'], self.images))

        key = lambda x: x
        if self.images and os.path.basename(self.images[0]).split('.')[0].isdigit():
            key = lambda x: int(os.path.basename(x).split('.')[0])

        self.images.sort(key=key)

        self.num = 0
        self.gray = gray

        firstImage = scio.loadmat(self.images[self.num])['img']

        self.height, self.weight = firstImage.shape[:2]

    def read(self):
        if self.num >= len(self):
            return False, None
        # img = cv2.imread(self.images[self.num], cv2.IMREAD_GRAYSCALE if self.gray else cv2.IMREAD_COLOR)
        img = scio.loadmat(self.images[self.num])['img']
        print(self.images[self.num])
        img = np.array(img, dtype=np.uint8)

        self.num += 1
        return True, img

    def get(self, idx):
        self.num = idx
        if self.num >= len(self):
            return False, None, None
        # img = cv2.imread(self.images[self.num], cv2.IMREAD_GRAYSCALE if self.gray else cv2.IMREAD_COLOR)
        img = scio.loadmat(self.images[self.num])['img']
        img = np.array(img, dtype=np.uint8)
        return True, img, os.path.basename(self.images[self.num])

    def getWeight(self):
        return self.weight

    def getHeight(self):
        return self.height

    def __len__(self):
        return len(self.images)

class imageCapture:
    def __init__(self, path, gray=False):
        self.path = path
        # self.images = sorted(glob.glob(os.path.join(self.path, '*.*g')))
        # self.images = sorted(glob.glob(os.path.join(self.path, '*.bmp')), key=lambda x: int(os.path.basename(x).split('.')[0]))

        self.images = glob.glob(os.path.join(self.path, '*.*'))
        self.images = list(filter(lambda x: os.path.basename(x).split('.')[1] in ['jpg', 'png', 'bmp'], self.images))

        key = lambda x: x
        if self.images and os.path.basename(self.images[0]).split('.')[0].isdigit():
            key = lambda x: int(os.path.basename(x).split('.')[0])

        self.images.sort(key=key)

        self.num = 0
        self.gray = gray

        firstImage = cv2.imread(self.images[0], cv2.IMREAD_GRAYSCALE if self.gray else cv2.IMREAD_COLOR)
        self.height, self.weight = firstImage.shape[:2]

    def read(self):
        if self.num >= len(self):
            return False, None
        img = cv2.imread(self.images[self.num], cv2.IMREAD_GRAYSCALE if self.gray else cv2.IMREAD_COLOR)
        self.num += 1
        return True, img

    def get(self, idx):
        self.num = idx
        if self.num >= len(self):
            return False, None, None
        img = cv2.imread(self.images[self.num], cv2.IMREAD_GRAYSCALE if self.gray else cv2.IMREAD_COLOR)
        return True, img, os.path.basename(self.images[self.num])

    def getWeight(self):
        return self.weight

    def getHeight(self):
        return self.height

    def __len__(self):
        return len(self.images)

class videoCapture:
    def __init__(self, path, gray=False):
        self.video = cv2.VideoCapture(path)
        self.gray = gray

    def read(self):
        ok, img = self.video.read()
        if ok and self.gray and len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return ok, img

    def getWeight(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))

    def getHeight(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __len__(self):
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

class Capture:
    def __init__(self, path, model=None, gray=True):
        if model is None:
            if os.path.basename(path)[-4:] in ['.mp4', '.avi', 'flv']:
                model = 'video'
            else:
                model = 'image'

        self.gray = gray

        if model == 'video':
            self.capture = videoCapture(path, gray)
        elif model == 'image':
            self.capture = imageCapture(path, gray)

    def read(self):
        return self.capture.read()

    def getHeight(self):
        return self.capture.getHeight()

    def getWeight(self):
        return self.capture.getWeight()

    def __len__(self):
        return len(self.capture)
