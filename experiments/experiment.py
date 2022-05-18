import os
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

class Experiment:
    def __init__(self, 
                 config: dict,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 generator_optimizer: optim.Optimizer,
                 discriminator_optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 dataloader: data.DataLoader) -> None:
        # setup the directories
        #TODO update to use kaggle path as well
        self.results_path = config['local_results_directory']
        self.experiment_name = config['experiment_name']
        self.full_path = f'{self.results_path}/{self.experiment_name}'
        # create paths
        if not os.path.isdir(self.results_path):
            os.mkdir(self.results_path)
        os.mkdir(self.full_path)
        # store models, optimizers, criterion, and data
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.criterion = criterion
        self.dataloader = dataloader
        # store data from config
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.noise_size = config['noise_size']
        self.save_checkpoint_every = config['save_checkpoint_every']
        self.save_image_every = config['save_image_every']
        # constants
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.fixed_noise = th.randn(64, self.noise_size, 1, 1, device=self.device)

    def train(self):
        for epoch in range(self.epochs):
            self.epoch()
            print(f'Epoch {epoch}: stats :D')
            if epoch % self.save_checkpoint_every:
                self.save_model_checkpoint(epoch)
            if epoch % self.save_image_every:
                self.save_model_image(epoch)

    def epoch(self):
        for (real, _) in self.dataloader:
            self.batch(real)

    def batch(self, real: th.Tensor):
        self.discriminator_optimizer.zero_grad()
        # discriminator on real images
        real = real.to(self.device)
        batch_size = real.size(0)
        true_labels = th.full((batch_size, ), 1, dtype=th.float, device=self.device)
        output = self.discriminator(real).view(-1)
        discriminator_real_error = self.criterion(output, true_labels)
        discriminator_real_error.backward()
        # discriminator on fake images
        noise = th.randn(batch_size, self.noise_size, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        fake_labels = th.full((batch_size, ), 0, dtype=th.float, device=self.device)
        output = self.discriminator(fake_images.detach()).view(-1)
        discriminator_fake_error = self.criterion(output, fake_labels)
        discriminator_fake_error.backward()
        # optimizer step
        self.discriminator_optimizer.step()
        # generator
        self.generator_optimizer.zero_grad()
        output = self.discriminator(fake_images).view(-1)
        generator_error = self.criterion(output, true_labels)
        generator_error.backward()
        self.generator_optimizer.step()

    def save_model_checkpoint(self, epoch: int) -> None:
        self.make_epoch_directories(epoch)
        checkpoint_path = f'self.full_path/{epoch}/checkpoint'
        if not os.path.isdir(checkpoint_path):
            os.mkdir(checkpoint_path)
        th.save({
            'epoch': epoch,
            'generator_model_state_dict': self.generator.state_dict(),
            'discriminator_model_state_dict': self.discriminator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
        })
        

    def save_model_image(self, epoch: int) -> None:
        self.make_epoch_directories(epoch)
        image_path = f'self.full_path/{epoch}/images'
        if not os.path.isdir(image_path):
            os.mkdir(image_path)
        random_noise = th.randn(64, self.noise_size, 1, 1, device=self.device)
        fixed_fakes = self.generator(self.fixed_noise).detach().cpu()
        random_fakes = self.generator(random_noise).detach().cpu()
        self.save_image_grid(fixed_fakes, f'{image_path}/fixed.png')
        self.save_image_grid(random_fakes, f'{image_path}/random.png')

    def save_image_grid(self, images, path: str) -> None:
        plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.title('Generated Images')
        plt.imshow(np.transpose(vutils.make_grid(images.to(self.device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.show()
        plt.savefig(path)
        plt.close()

    def make_epoch_directories(self, epoch: int) -> None:
        epoch_path = f'self.full_path/{epoch}'
        if not os.path.isdir(epoch_path):
            os.mkdir(epoch_path)