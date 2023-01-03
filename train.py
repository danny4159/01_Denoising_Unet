from torch.utils.data import DataLoader
import torch
import argparse
from dataset import ImageDataset, NoiseImageDataset 
import torch.nn as nn
from UNet import UNet
import torchvision.transforms as transforms
import pdb # pdb.set_trace() // 커맨드: n, d, q
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0,2,6,7"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Train():
    def __init__(self):
        self.EPOCHS, self.BATCH, self.STD, self.LR, self.DATA_DIR, self.CH_DIR = self.__get_args__()
        self.transforms = transforms.Compose([
                                              transforms.CenterCrop(448),
                                              transforms.ToTensor(),
                                            #   transforms.Normalize((0.5,), (0.5,)) # TODO: 이 normalize를 하니까 결과 이미지를 뽑을 때 어두워지고 처리가 어려워져
                                              ])
        self.trainloader, self.validloader = self.load_data()
        self.use_cuda = torch.cuda.is_available()
        
        
    def __get_args__(self):
        parser = argparse.ArgumentParser(description='Parameters')
        parser.add_argument('--epochs', type=int, default=30)
        parser.add_argument('--batch', type=int, default=8)
        parser.add_argument('--std', type=float, default=0.1, help='Standard Deviation on Gaussian Noise')
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--data_dir', type=str, default='./data')
        parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
        args = parser.parse_args()
        return args.epochs, args.batch, args.std, args.learning_rate, args.data_dir, args.checkpoint_dir
        
    def load_data(self):
        train_dataset = ImageDataset(img_dir = self.DATA_DIR + '/train/', transform=self.transforms)
        valid_dataset = ImageDataset(img_dir = self.DATA_DIR + '/val/', transform=self.transforms)
        train_dataloader = DataLoader(train_dataset, batch_size=self.BATCH, num_workers=4, shuffle=True) # shuffle: 데이터 섞어서 과적합 방지
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.BATCH, num_workers=4, shuffle=True)
        return train_dataloader, valid_dataloader
    
    def get_model(self):
        model = UNet(in_channels=1, out_channels=1).double()
        if self.use_cuda:
            model = nn.DataParallel(model).to(device)
        noisy_dataset = NoiseImageDataset(std=self.STD)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LR)
        criterion = MSELoss()
        return model, noisy_dataset, optimizer, criterion
        
    def train(self):
        model, noisy_dataset, optimizer, criterion = self.get_model()
        
        train_loss =[]
        val_loss= []
        epochs=[]    
        min_val_loss = 100
        for epoch in range(self.EPOCHS):
            loss_epoch = 0
            loss = 0        
            for idx, batch in enumerate(self.trainloader): # _는 들어오는거 무시하는 것
                optimizer.zero_grad()
                image = batch
                noisy_image = noisy_dataset(batch)
                if self.use_cuda:
                    image = image.to(device)
                    noisy_image = noisy_image.to(device)
                denoised_image = model(noisy_image)
                loss = criterion(denoised_image,image)
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
            train_loss_avg = loss_epoch / len(self.trainloader)
            train_loss.append(train_loss_avg)
            
            loss_epoch = 0
            loss_valid = 0
            for idx, batch in enumerate(self.validloader):
                with torch.no_grad():
                    image = batch
                    noisy_image = noisy_dataset(batch)
                    if self.use_cuda:
                        image = image.to(device)
                        noisy_image = noisy_image.to(device)
                    denoised_image = model(noisy_image)
                    loss_valid = criterion(denoised_image,image)
                    loss_epoch += loss_valid.item()
                    
            val_loss_avg = loss_epoch / len(self.validloader)        
            val_loss.append(val_loss_avg)
            epochs.append(epoch+1)
            print("Epoch: " + str(epoch+1) + " train_loss: " + str(train_loss_avg) + " val_loss: " + str(val_loss_avg))
                    
            if val_loss_avg < min_val_loss:
                min_val_loss = val_loss_avg
                torch.save({
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch':epoch+1
                            }, self.CH_DIR + '/chk_1_std_' + str(self.STD) + '.pt') # TODO: 저장할때마다 이름 바꿔주기
                print('Saving Model...')
            
        plt.plot(epochs, train_loss, label="train loss", color="red",linestyle=':')
        plt.plot(epochs, val_loss, label="val loss", color="green", linestyle=':')
        plt.title("Loss Curve")
        plt.legend()
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()
        plt.savefig('Loss Curve.png')
            
            
class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def mseloss(self, image, target): # 편차 제곱의 평균
        x = (image - target)**2 
        return torch.mean(x)
        
    def forward(self, image, target):
        return self.mseloss(image, target)
    

if __name__ == '__main__': # 이 py파일을 main으로 쓸때만 실행
    TrainUnet = Train()
    TrainUnet.train()
                