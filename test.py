import torch
from UNet import UNet
from dataset import ImageDataset, NoiseImageDataset
import argparse
from torchvision.utils import save_image
from train import Train
from torchvision import transforms

class Test():
    def __init__(self):
        self.STD, self.DATA_DIR, self.CHECKPOINT = self.__get_args__()
        self.transforms = Train().transforms
        
    def __get_args__(self):
        parser = argparse.ArgumentParser(description='Parameters')
        parser.add_argument('--std', type=float, default=0.1)
        parser.add_argument('--data_dir', type=str, default='./data/test')
        parser.add_argument('--checkpoint', type=str,
                            default='./checkpoints/chk_16_std_0.1.pt') # TODO: 원하는 checkpoint 이름으로 수정
        args = parser.parse_args()
        return args.std, args.data_dir, args.checkpoint

    def img_denorm(self, img, mean, std): # Denormalize: Normalize를 이전의 상태로
        denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))
        res = denormalize(img)
        res = torch.clip(res, 0, 1)
        return(res)

    def test(self):          
        testset = ImageDataset(self.DATA_DIR, transform=self.transforms)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4)
        
        checkpoint = torch.load(self.CHECKPOINT, map_location=torch.device('cpu'))
        model_test = UNet(in_channels=1, out_channels=1).double()
        model_test.load_state_dict(checkpoint['model_state_dict'])
        model_test = model_test.cpu()
        model_test.train()
            
        for idx, batch in enumerate(testloader):
            images = batch
            noisy = NoiseImageDataset(std=self.STD)
            noisy_images = noisy(images)

            ### 이미지 저장
            for i in range((images.cpu()).size()[0]): # batch 개수 파악
                
                ## 1.원본 이미지 저장
                image = images.cpu()[i,:,:]
                image = self.img_denorm(image,0.5,0.5)
                # print("origin max: " + str(image.max()))
                # print("origin min: " + str(image.min()))
                save_image(image,'./results/original_image_batch_'+str(idx)+"_"+str(i)+'.jpg')
                
                ## 2.노이즈 이미지 저장
                noisy_image = noisy_images.cpu()[i,:,:]
                noisy_image = self.img_denorm(noisy_image,0.5,0.5)
                # print("noisy_image max: " + str(noisy_image.max()))
                # print("noisy_image min: " + str(noisy_image.min()))
                save_image(noisy_image,'./results/noisy_image_batch_'+str(idx)+"_"+str(i)+'.jpg')
                
                ## 3.디노이즈 이미지 저장
                denoisy_image = model_test(noisy_images.cpu())[i,:,:,:]
                denoisy_image = self.img_denorm(denoisy_image,0.5,0.5)
                # print("denoisy_image max: " + str(denoisy_image.max()))
                # print("denoisy_image min: " + str(denoisy_image.min()))
                save_image(denoisy_image,'./results/denoised_image_batch_'+str(idx)+"_"+str(i)+'.jpg')
            print("Batch " + str(idx) + " Done...")
        print('End')
    
if __name__ == '__main__': # 이 py파일을 main으로 쓸때만 실행
    TestUnet = Test()
    TestUnet.test()
                