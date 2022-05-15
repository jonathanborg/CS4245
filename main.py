import torch as th
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np


from models import Generator, Discriminator
from utils import weights_init, save_images

dataroot = './data/faces'
batch_size = 128
image_size = 64
nc = 3
nz = 100
ngf = 64
ndf = 64
num_epochs = 5
lr = 0.0002
beta1 = 0.5
ngpu = 1

device = th.device('cuda:0' if th.cuda.is_available() and ngpu > 0 else 'cpu')
transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
dataset = torchvision.datasets.ImageFolder(dataroot, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

batch = next(iter(dataloader))
save_images(batch, device, 'real.png')


netG = Generator(nz, nc, ngf).to(device)
netG.apply(weights_init)
print(netG)
netD = Discriminator(nc, ndf).to(device)
netD.apply(weights_init)
print(netD)

criterion = th.nn.BCELoss()
fixed_noise = th.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0
optimizerD = th.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = th.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

fake = netG(fixed_noise).detach()
discriminated = netD(fake).detach()
print(f'Image size size: {batch[0][0].size()}')
print(f'Generated size: {fake[0].size()}')
print(f'Discriminated size: {discriminated[0].size()}')


for epoch in range(num_epochs):
    for (real, _) in tqdm(dataloader):
        netD.zero_grad()
        real = real.to(device)
        b_size = real.size(0)
        label = th.full((b_size, ), real_label, dtype=th.float, device=device)
        output = netD(real).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = th.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimizerD.step()


        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

    with th.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.title('Training images')
        plt.imshow(np.transpose(vutils.make_grid(fake.to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.show()
        plt.savefig(f'./results/generated-{epoch}.png')
        plt.close()