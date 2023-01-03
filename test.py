import torch
from UNet import UNet
from dataset import ImageDataset, NoiseImageDataset
import argparse
from torchvision.utils import save_image
from train import Train

class Test():
    def __init__(self):
        self.STD, self.DATA_DIR, self.CHECKPOINT = self.__get_args__()
        self.transforms = Train().transforms
        
    def __get_args__(self):
        parser = argparse.ArgumentParser(description='Parameters')
        parser.add_argument('--std', type=float, default=0.1)
        parser.add_argument('--data_dir', type=str, default='./data/test')
        parser.add_argument('--checkpoint', type=str,
                            default='./checkpoints/chk_1_std_0.1.pt')
        args = parser.parse_args()
        return args.std, args.data_dir, args.checkpoint

    def test(self):          
        testset = ImageDataset(self.DATA_DIR, transform=self.transforms)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4)
        dataiter = iter(testloader)
        checkpoint = torch.load(self.CHECKPOINT, map_location=torch.device('cpu'))
        
        model_test = UNet(in_channels=1, out_channels=1).double()
        model_test.load_state_dict(checkpoint['model_state_dict'])
        model_test = model_test.cpu()
        model_test.train()
        images = next(dataiter)
        noisy = NoiseImageDataset(std=self.STD)
        noisy_images = noisy(images)
        print((images.cpu()).size())

        ## 이미지 저장
        for i in range((images.cpu()).size()[0]): # batch 개수 파악
            # 원본 이미지 저장
            image = images.cpu()[i,:,:]
            save_image(image,'./results/original_image'+str(i)+'.jpg')
            # 노이즈 이미지 저장
            noisy_image = noisy_images.cpu()[i,:,:]
            save_image(noisy_image,'./results/noisy_image'+str(i)+'.jpg')
            # 디노이즈 이미지 저장
            denoisy_image = model_test(noisy_images.cpu())[i,:,:,:]
            save_image(denoisy_image,'./results/denoised_image'+str(i)+'.jpg')
        
        print('end')
    
if __name__ == '__main__': # 이 py파일을 main으로 쓸때만 실행
    TestUnet = Test()
    TestUnet.test()
                