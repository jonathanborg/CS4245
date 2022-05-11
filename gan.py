import torch as th
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from tqdm import tqdm

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
        torchvision.transforms.Resize(64),
        torchvision.transforms.ToTensor()
    ])
dataset = torchvision.datasets.ImageFolder(dataroot, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis('off')
plt.title('Training images')
plt.imshow(np.transpose(vutils.make_grid(batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()
plt.savefig('./generator_results/real.png')
plt.close()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        th.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        th.nn.init.normal_(m.weight.data, 1.0, 0.02)
        th.nn.init.constant_(m.bias.data, 0)

class Generator(th.nn.Module):
    def __init__(self, ngpu) -> None:
        super().__init__()
        self.ngpu = ngpu
        self.main = th.nn.Sequential(
            # input is z
            th.nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            th.nn.BatchNorm2d(ngf * 8),
            th.nn.ReLU(True),
            # state size (ngf*8) * 4 * 4
            th.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            th.nn.BatchNorm2d(ngf * 4),
            th.nn.ReLU(True),
            # state size (ngf * 4) * 8 * 8
            th.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            th.nn.BatchNorm2d(ngf * 2),
            th.nn.ReLU(True),
            # state size (ngf * 2) * 16 * 16
            th.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            th.nn.BatchNorm2d(ngf),
            th.nn.ReLU(True),
            # state size ngf * 32 * 32,
            th.nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            th.nn.Tanh()   
        )

    def forward(self, x):
        return self.main(x)


class Discriminator(th.nn.Module):
    def __init__(self, ngpu) -> None:
        super().__init__()
        self.ngpu = ngpu
        self.main = th.nn.Sequential(
            th.nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            th.nn.LeakyReLU(0.2, inplace=True),

            th.nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            th.nn.BatchNorm2d(ndf * 2),
            th.nn.LeakyReLU(0.2, inplace=True),

            th.nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            th.nn.BatchNorm2d(ndf * 4),
            th.nn.LeakyReLU(0.2, inplace=True),

            th.nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            th.nn.BatchNorm2d(ndf * 8),
            th.nn.LeakyReLU(0.2, inplace=True),

            th.nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            th.nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x)


netG = Generator(ngpu).to(device)
netG.apply(weights_init)
print(netG)
netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
print(netD)

criterion = th.nn.BCELoss()
fixed_noise = th.randn(64, nz, 1, 1, device=device)
real_label = 1
fake_label = 0
optimizerD = th.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = th.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


img_list = []
G_losses = []
D_losses = []
iters = 0

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
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()


        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
        #           % (epoch, num_epochs, i, len(dataloader),
        #              errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

    with th.no_grad():
        fake = netG(fixed_noise).detach().cpu()
        plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.title('Training images')
        plt.imshow(np.transpose(vutils.make_grid(fake.to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.show()
        plt.savefig(f'./generator_results/generated-{epoch}.png')
        plt.close()

with th.no_grad():
    fake = netG(fixed_noise).detach().cpu()
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title('Training images')
    plt.imshow(np.transpose(vutils.make_grid(fake.to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
    plt.savefig('./generator_results/generated_final.png')
    plt.close()