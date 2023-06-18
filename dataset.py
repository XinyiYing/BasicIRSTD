from utils import *
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

IMG_EXTENSIONS = ('.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm',
                  '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF')

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        super(TrainSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + dataset_name
        self.patch_size = patch_size
        with open(self.dataset_dir+'/img_idx/train_' + dataset_name + '.txt', 'r') as f:
            self.train_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
        
    def __getitem__(self, idx):
        img_list = os.listdir(self.dataset_dir + '/images/')
        img_ext = os.path.splitext(img_list[0])[-1]
        if not img_ext in IMG_EXTENSIONS:
            raise TypeError("Unrecognized image extensions.")
        img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + img_ext).replace('//','/')).convert('I')
        mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + img_ext).replace('//','/'))
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape)>3:
            mask = mask[:,:,0]
        
        img_patch, mask_patch = random_crop(img, mask, self.patch_size)
        img_patch, mask_patch = self.tranform(img_patch, mask_patch)
        img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))
        return img_patch, mask_patch
    def __len__(self):
        return len(self.train_list)

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir+'/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        img_list = os.listdir(self.dataset_dir + '/images/')
        img_ext = os.path.splitext(img_list[0])[-1]
        if not img_ext in IMG_EXTENSIONS:
            raise TypeError("Unrecognized image extensions.")
        img = Image.open((self.dataset_dir + '/images/' + self.train_list[idx] + img_ext).replace('//','/')).convert('I')
        mask = Image.open((self.dataset_dir + '/masks/' + self.train_list[idx] + img_ext).replace('//','/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape)>3:
            mask = mask[:,:,0]
            
        h, w = img.shape
        img = np.pad(img, ((0, (h//32+1)*32-h),(0, (w//32+1)*32-w)), mode='constant')
        mask = np.pad(mask, ((0,(h//32+1)*32-h),(0,(w//32+1)*32-w)), mode='constant')
        
        img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 

class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        with open(self.dataset_dir+'/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        img_list_pred = os.listdir(self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/')
        img_ext_pred = os.path.splitext(img_list[0])[-1]

        img_list_gt = os.listdir(self.dataset_dir + '/masks/')
        img_ext_gt = os.path.splitext(img_list[0])[-1]
        
        if not img_ext in IMG_EXTENSIONS:
            raise TypeError("Unrecognized image extensions.")
        mask_pred = Image.open((self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' + self.test_list[idx] + img_ext_pred).replace('//','/'))
        mask_gt = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + img_ext_gt).replace('//','/'))
                
        mask_pred = np.array(mask_pred, dtype=np.float32)  / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32)  / 255.0
        if len(mask_pred.shape)>3:
            mask_pred = mask_pred[:,:,0]
        if len(mask_gt.shape)>3:
            mask_gt = mask_gt[:,:,0]
            
        h, w = mask_pred.shape
        
        mask_pred, mask_gt = mask_pred[np.newaxis,:], mask_gt[np.newaxis,:]
        
        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h,w]
    def __len__(self):
        return len(self.test_list) 


class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input, target
