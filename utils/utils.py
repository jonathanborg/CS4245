import torch as th
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
import os
import csv

def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        th.nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        th.nn.init.normal_(model.weight.data, 1.0, 0.02)
        th.nn.init.constant_(model.bias.data, 0)


class SaveModel:
    def __init__(self, generator, discriminator, generator_optimizer, discriminator_optimizer, full_path, noise_size, device, fixed_noise):
        self.full_path = full_path
        self.noise_size = noise_size
        self.device = device
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.fixed_noise = fixed_noise

    def save_model_checkpoint(self, epoch: int) -> None:
        self.make_epoch_directories(epoch)
        checkpoint_path = f'{self.full_path}/{epoch}'
        th.save({
            'epoch': epoch,
            'generator_model_state_dict': self.generator.state_dict(),
            'discriminator_model_state_dict': self.discriminator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
        }, f'{checkpoint_path}/checkpoint.th')

    def save_model_image(self, epoch: int) -> None:
        self.make_epoch_directories(epoch)
        image_path = f'{self.full_path}/{epoch}/images'
        if not os.path.isdir(image_path):
            os.mkdir(image_path)
        random_noise = th.randn(64, self.noise_size, 1, 1, device=self.device)
        fixed_fakes = self.generator(self.fixed_noise).detach().cpu()
        random_fakes = self.generator(random_noise).detach().cpu()
        self.save_image_grid(fixed_fakes, f'{image_path}/fixed.png', 'Fixed Noise')
        self.save_image_grid(random_fakes, f'{image_path}/random.png', 'Random Noise')

    def save_model_metrics(self, epoch: int, model_metrics) -> None:
        self.make_epoch_directories(epoch)
        metrics_path = f'{self.full_path}/{epoch}/metrics.csv'
        with open(metrics_path, 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(model_metrics.keys())
            writer.writerows(zip(*model_metrics.values()))

    def save_image_grid(self, images, path: str, title: str) -> None:
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.title(title)
        plt.imshow(
            np.transpose(vutils.make_grid(images.to(self.device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.savefig(path)
        plt.close()

    def make_epoch_directories(self, epoch: int) -> None:
        epoch_path = f'{self.full_path}/{epoch}'
        if not os.path.isdir(epoch_path):
            os.mkdir(epoch_path)


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

