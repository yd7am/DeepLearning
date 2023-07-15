# 数据预处理
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from os.path import join
from os import listdir
from PIL import Image
import torch

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir):
        super(TrainDatasetFromFolder, self).__init__()
        self.images_filenames = [join(dataset_dir,x) for x in listdir(dataset_dir)]
        self.transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])
    
    def __getitem__(self, index):
        real_image = self.transform(Image.open(self.images_filenames[index]))
        return real_image, torch.Tensor([1])
    
    def __len__(self):
        return len(self.images_filenames)
    
dsets_paths = "../../datasets/face/images"
dsets = TrainDatasetFromFolder(dsets_paths)
dlsets = DataLoader(dsets, batch_size=64, shuffle=True, num_workers=4)

# 模型构建
import torch
from torch import nn

# 生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.g = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.weight_init(0.0, 0.02)
        
    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(mean, std)
    
    def forward(self, x):
        return self.g(x)
    
# 判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.d = nn.Sequential(
            nn.Conv2d(3, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 4, 1, 0)
        )
        self.weight_init(0.0, 0.02)
        
    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
    
    def forward(self, x):
        return self.d(x)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# training parameters
learning_rate = 0.00005
epochs = 1000
weight_clip = 0.01
# network
G = Generator().to(device)
D = Discriminator().to(device)
G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

import matplotlib.pyplot as plt

def show_result(z, epoch, size_figure_grid=3):
    G.eval()
    test_images = G(z)
    G.train()
    
    plt.figure(figsize=(size_figure_grid,size_figure_grid)) 
    for i in range(size_figure_grid**2):
        img = test_images[i]
        plt.subplot(size_figure_grid,size_figure_grid,i+1)
        plt.imshow((img.data.permute(1,2,0).cpu().numpy() + 1) / 2)
        plt.axis('off')

    plt.savefig(f'./results/{epoch+1}.png')
    plt.close('all')

from tqdm import tqdm
import numpy as np
import time
import imageio

results = {'d_loss': [], 'g_loss': []}
size_figure_grid = 7
fix_z = torch.randn((size_figure_grid**2, 100, 1, 1), device=device)
for epoch in range(epochs):
    running_results = {'d_loss': [], 'g_loss': []}
    G.train()
    D.train()
    epoch_start_time = time.time()
    for ii, (pr_images, _) in tqdm(enumerate(dlsets), total=len(dlsets)):
        batch_size = pr_images.shape[0]
        pr_images = pr_images.to(device)
        # train D
        for p in D.parameters():
            p.requires_grad = True
        
        # Clamp parameters to a range [-c, c], c=weight_clip
        # 通过在训练过程中保证判别器的所有参数有界，就保证了判别器不能
        # 对两个略微不同的样本在判别上不会差异过大，从而间接实现了Lipschitz限制。
        for parm in D.parameters():
            parm.data.clamp_(-weight_clip,weight_clip)
            
        D.zero_grad()
        z = torch.randn((batch_size, 100, 1, 1), device=device)
        pg_images = G(z)
        loss_d = D(pg_images).mean() - D(pr_images).mean()  # 不取log
        loss_d.backward()
        D_optimizer.step()
        # train G
        for p in D.parameters():
            p.requires_grad = False

        G.zero_grad()
        z = torch.randn((batch_size, 100, 1, 1), device=device)
        pg_images = G(z)
        loss_g = -D(pg_images).mean()
        loss_g.backward()
        G_optimizer.step()
        
        running_results['d_loss'].append(loss_d.item())
        running_results['g_loss'].append(loss_g.item())
        
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    epoch_d_loss = np.mean(running_results['d_loss'])
    epoch_g_loss = np.mean(running_results['g_loss'])
    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % (epoch + 1, epochs, per_epoch_ptime, epoch_d_loss, epoch_g_loss))
    results['d_loss'].append(epoch_d_loss)
    results['g_loss'].append(epoch_g_loss)
    show_result(fix_z, epoch, size_figure_grid)

plt.plot(range(epochs), results['d_loss'], label='d_loss')
plt.plot(range(epochs), results['g_loss'], label='g_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('Train_hist.png')