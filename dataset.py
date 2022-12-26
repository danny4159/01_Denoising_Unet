
from torch.utils.data import Dataset
import os
from PIL import Image
import torch.nn as nn
import numpy as np

class ImageDataset(Dataset):
    ### 데이터셋 전처리 - 입력: img_dir     
    def __init__(self, img_dir): # TODO: transform 뺏어 추후 쓸 때 추가하기
        self.img_dir = img_dir
        
    ### 데이터셋 길이 - 해당 폴더에 있는 이미지 파일 개수 세기
    def __len__(self):
        img_list = os.listdir(self.img_dir)
        return len(img_list)
    
    ### 데이터셋에서 특정 1개 샘플 가져오기
    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir, str(idx) + '.jpg')
        img = Image.open(img_path) # TODO: convert('RGB') 안했어 뭐가 다를까
        return img


class NoiseImageDataset(nn.Module):
    def __init__(self, mean = 0, std = 1, img=None):  # TODO: rootdir를 뺐어. 나중에 어떻게 쓰이려나? # TODO: std를 조절해서 더 큰 noise도 만들어보기
        super(NoiseImageDataset, self).__init__()
        self.mean = mean
        self.std = std
        self.img = img
        
    def makeGaussianNoise(self, h, w, ch):
        # 참고: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
        img_noise = np.zeros(shape=self.img.shape, dtype=np.float64)
        gaussian_noise = np.random.normal(loc = self.mean, scale = self.std, size = (h,w,ch))
        img_noise = self.img + gaussian_noise

        return img_noise
           
    def avgGaussianNoise(self):
        h, w, ch = self.img.shape
        
        # 가우시안 노이즈 결과 4개
        img_noise = self.makeGaussianNoise(h, w, ch)  # 클래스 안의 메소드 부를 때 self 붙이는구나!
        img_noise2 = self.makeGaussianNoise(h, w, ch)
        img_noise3 = self.makeGaussianNoise(h, w, ch)
        img_noise4 = self.makeGaussianNoise(h, w, ch)
 
        # 4개 결과를 평균
        img_noise_avg = np.zeros((h, w, ch), dtype=np.float64)
        for i in range(h):
            for j in range(w):
                for k in range(ch):
                    if(img_noise[i,j,k] + img_noise2[i,j,k] + img_noise3[i,j,k] + img_noise4[i,j,k]) / 4 > 255:
                        img_noise_avg[i,j,k] = 255
                    else:
                        img_noise_avg[i,j,k] = (img_noise[i,j,k] + img_noise2[i,j,k] + img_noise3[i,j,k] + img_noise4[i,j,k]) / 4
                    
        return img_noise_avg