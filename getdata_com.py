from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torchvision.transforms as transform
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2


SUIM_CLASSES = ( 
                'Human',
               'Robot',
               'Fish',
               'Reef',
               'Wreck')

NUM_CLASSES = len(SUIM_CLASSES) 

class combineNet(Dataset):
    
    def __init__(self, img_dir, mask_dir, image_shape=(224, 224), n_channels=3, transform=None):
        #self.images = open(list_file, "rt").read().split("\n")[:-1]
        
        self.imgshape = image_shape
        self.n_channels = n_channels
        self.img_extension = ".jpg"
        self.mask_extension = ".bmp"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir
        self.transform = transform

        #self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(os.listdir(self.image_root_dir))

    def __getitem__(self, index):
        image_name = os.listdir(self.image_root_dir)[index]
        name = image_name.split('.')[0]
        image_path = os.path.join(self.image_root_dir, name + self.img_extension)
        mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

        image = self.load_image(path=image_path)/255.
        mask = self.load_mask(path=mask_path)
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].permute(2,0,1).type(torch.float32)
            post_image = torch.cat((image,mask),0)
            #import ipdb; ipdb.set_trace()
        data = {
                    'prior_image': image,
                    'post_image': post_image,
                    'mask' : mask
                    }
        #import ipdb; ipdb.set_trace()
        return data
        
    def load_image(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize(self.imgshape, Image.BILINEAR)
        imx_t = np.array(raw_image, dtype=np.float32)
        return imx_t
    
    def load_test(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize(self.imgshape, Image.BILINEAR)
        imx_t = np.array(raw_image, dtype=np.float32)
        
        imx_t = self.transform(imx_t)
        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize(self.imgshape, Image.BILINEAR)
        raw_image = np.array(raw_image)
        raw_image = raw_image/255.0
        raw_image[raw_image > 0.5] = 1
        raw_image[raw_image <= 0.5] = 0
        mask = []
        mask.append(self.getDiffMask(raw_image))
        imx_t = np.array(mask).squeeze()
        # print(imx_t.shape)
        # border
        #imx_t[imx_t==255] = len(VOC_CLASSES)

        return imx_t

    def getDiffMask(self, mask):
        imw, imh = mask.shape[0], mask.shape[1]

        
        #################
        # softmax
        #################
        background = np.ones((imw, imh))
        Human1 = np.zeros((imw, imh))
        Robot1 = np.zeros((imw, imh))
        Fish1 = np.zeros((imw, imh))
        Reef1 = np.zeros((imw, imh))
        Wreck1 = np.zeros((imw, imh))

        mask_idx = np.where((mask[:,:,0] == 0) & (mask[:,:,1] == 0) & (mask[:,:,2] == 1))
        Human1[mask_idx] = 1
        background[mask_idx] = 0
    
        mask_idx = np.where((mask[:,:,0] == 1) & (mask[:,:,1] == 0) & (mask[:,:,2] == 0))
        Robot1[mask_idx] = 1
        background[mask_idx] = 0

        mask_idx = np.where((mask[:,:,0] == 1) & (mask[:,:,1] == 1) & (mask[:,:,2] == 0))
        Fish1[mask_idx] = 1
        background[mask_idx] = 0

        mask_idx = np.where((mask[:,:,0] == 1) & (mask[:,:,1] == 0) & (mask[:,:,2] == 1))
        Reef1[mask_idx] = 1
        background[mask_idx] = 0

        mask_idx = np.where((mask[:,:,0] == 0) & (mask[:,:,1] == 1) & (mask[:,:,2] == 1))
        Wreck1[mask_idx] = 1
        background[mask_idx] = 0
        
        return np.stack((Robot1, Fish1, Human1, Reef1, Wreck1), -1) 
