
from torch.utils.data import Dataset
import os
from PIL import Image

class ImageDataset(Dataset):
    ### 데이터셋 전처리 - 입력: img_dir     
    def __init__(self, img_dir):
        self.img_dir = img_dir
        
    ### 데이터셋 길이 - 해당 폴더에 있는 이미지 파일 개수 세기
    def __len__(self):
        img_list = os.listdir(self.img_dir)
        return len(img_list)
    
    ### 데이터셋에서 특정 1개 샘플 가져오기
    # TODO: 추후 활용할 때 코드 수정해야해
    def __getitem__(self,idx):
        img = Image.open(self.img_dir)
        return img

        