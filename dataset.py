from torch.utils.data import Dataset
import os
from PIL import Image
import torch.nn as nn
import numpy as np
import torch
from torchvision.utils import save_image

class ImageDataset():
    def __init__(self, img_dir = None, transform = None):
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        img_list = os.listdir(self.img_dir)
        return len(img_list)
    
    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir, str(idx) + '.jpg')
        img = Image.open(img_path).convert('L') # 흑백이미지만 받아. 컬러이미지는 코드 수정 필요.
        if self.transform:
            img = self.transform(img)
        return img


class NoiseImageDataset(nn.Module):
    def __init__(self, mean = 0, std = 1): # , img = None  
        super(NoiseImageDataset, self).__init__()
        self.mean = mean
        self.std = std
        
    def makeGaussianNoise(self):
        # 참고: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
        img_noise = np.zeros(shape=self.img.shape, dtype=np.float64)
        gaussian_noise = np.random.normal(loc = self.mean, scale = self.std, size = self.img.shape)
        img_noise = self.img + gaussian_noise
        return img_noise
            
    def forward(self, img):
        self.img = img  # train.py에서 객체 한개만 생성하여 노이즈 양산하기 위해 forward에서 선언 
        img_noise = self.makeGaussianNoise()
        img_noise = np.clip(img_noise, 0, 1) # clip: 0이하는 0으로 1이상은 1로  # TODO: 0에서 1 clip말고 더 좋은 방법이 있을까. 
        img_noise = img_noise.clone().detach() # numpy -> tensor
        return img_noise