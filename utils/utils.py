import torch as th
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import os


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        th.nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        th.nn.init.normal_(model.weight.data, 1.0, 0.02)
        th.nn.init.constant_(model.bias.data, 0)


def save_images(images, device, name, path='./results'):
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title('Images')
    plt.imshow(np.transpose(vutils.make_grid(images[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig(f'{path}/{name}')
    plt.close()


def save_model_snapshot(netD, netG, epoch, fixed_noise, device, nz):
    path = f'./results/{epoch}'
    os.mkdir(path)
    noise = th.randn(64, nz, 1, 1, device=device)
    fixed_fake = netG(fixed_noise).detach().cpu()
    fake = netG(noise).detach().cpu()
    save_images(fixed_fake, device, f'fixed-generated-{epoch}.png', path=path)
    save_images(fake, device, f'random-generated-{epoch}.png', path=path)
    th.save(netD.state_dict(), f'{path}/netD.pth')
    th.save(netG.state_dict(), f'{path}/netG.pth')

