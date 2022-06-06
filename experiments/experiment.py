import os
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from tqdm import tqdm

import experiments.evaluation as evaluation
import csv

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
        self.evaluation = config['evaluation']
        
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
        self.save_metrics_every = config['save_metrics_every']
        self.true_label_value = config['true_label_value']
        self.fake_label_value = config['fake_label_value']

        # constants
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.fixed_noise = th.randn(64, self.noise_size, 1, 1, device=self.device)
        self.datasize = len(self.dataloader.dataset)

        self.model_metrics = {
            'epochs': [],
            'fid': [],
            'loss_g': [],
            'loss_d_real': [],
            'loss_d_fake': [],
            'accuracy_g': [],
            'accuracy_d_real': [],
            'accuracy_d_fake': []
        }


    def train(self):
        for epoch in range(self.epochs):
            total_fid, total_real_error, total_fake_error, total_generator_error, total_real_correct, total_fake_correct, total_generator_correct = self.epoch()
            # metric calculation
            real_accuracy = total_fake_correct / self.datasize
            fake_accuracy = total_real_correct / self.datasize
            generator_accuracy = total_generator_correct / self.datasize
            fid = total_fid / self.datasize
            real_error = total_real_error / self.datasize
            fake_error = total_fake_error / self.datasize
            generator_error = total_generator_error / self.datasize

            loss_string = f'Loss_G: {generator_error:.4f}\tReal_Loss_D: {real_error:.4f}\tFake_Loss_D: {fake_error:.4f}'
            accuracy_string = f'Accuracy_G: {generator_accuracy:.4f}\tReal_Accuracy_D: {real_accuracy:.4f}\tFake_Accuracy_D: {fake_accuracy:.4f}'
            print(f'{epoch+1}/{self.epochs}: FID: {fid:.4f}\t{loss_string}\t{accuracy_string}')

            self.model_metrics['epochs'].append(epoch)
            self.model_metrics['fid'].append(fid)
            self.model_metrics['loss_g'].append(generator_error)
            self.model_metrics['loss_d_real'].append(real_error)
            self.model_metrics['loss_d_fake'].append(fake_error)
            self.model_metrics['accuracy_g'].append(generator_accuracy)
            self.model_metrics['accuracy_d_real'].append(real_accuracy)
            self.model_metrics['accuracy_d_fake'].append(fake_accuracy)

            with th.no_grad():
                if epoch % self.save_checkpoint_every == 0:
                    print('-> Saving model checkpoint')
                    self.save_model_checkpoint(epoch)
                if epoch % self.save_image_every == 0:
                    print('-> Saving model images')
                    self.save_model_image(epoch)
                if epoch % self.save_metrics_every == 0:
                    print('-> Saving metrics')
                    self.save_model_metrics(epoch)

    def epoch(self):
        total_fid = 0
        total_real_error = 0
        total_fake_error = 0
        total_generator_error = 0
        total_real_correct = 0
        total_fake_correct = 0
        total_generator_correct= 0 
        for (real, _) in tqdm(self.dataloader, leave=False):
            real_error, fake_error, generator_error, fid, real_correct, fake_correct, generator_correct = self.batch(real)

            total_fid += fid
            total_real_error += real_error
            total_fake_error += fake_error
            total_generator_error += generator_error
            total_real_correct += real_correct
            total_fake_correct += fake_correct
            total_generator_correct += generator_correct

        return total_fid, total_real_error, total_fake_error, total_generator_error, total_real_correct, total_fake_correct, total_generator_correct

    def batch(self, real_images: th.Tensor):
        self.discriminator_optimizer.zero_grad()

        # discriminator on real images
        real_images = real_images.to(self.device)
        batch_size = real_images.size(0)
        real_labels = th.full((batch_size, ), self.true_label_value, dtype=th.float, device=self.device)
        real_predicted = self.discriminator(real_images).view(-1)
        # print(real_predicted)
        discriminator_real_error = self.criterion(real_predicted, real_labels)
        discriminator_real_error.backward()

        # discriminator on fake images
        noise = th.randn(batch_size, self.noise_size, 1, 1, device=self.device)
        fake_images = self.generator(noise)
        fake_labels = th.full((batch_size, ), self.fake_label_value, dtype=th.float, device=self.device)
        fake_predicted = self.discriminator(fake_images.detach()).view(-1)
        # print(fake_predicted)
        discriminator_fake_error = self.criterion(fake_predicted, fake_labels)
        discriminator_fake_error.backward()
        # discrminiator optimizer step
        self.discriminator_optimizer.step()

        # generator
        self.generator_optimizer.zero_grad()
        generator_fake_predicted = self.discriminator(fake_images).view(-1)
        # print(generator_fake_predicted)
        generator_error = self.criterion(generator_fake_predicted, real_labels)
        generator_error.backward()
        self.generator_optimizer.step()
        
        fid, real_correct, fake_correct, generator_correct = self.metrics(real_images, fake_images, real_labels, fake_labels, real_predicted, fake_predicted, generator_fake_predicted)
        return discriminator_real_error.item(), discriminator_fake_error.item(), generator_error.item(), fid, real_correct, fake_correct, generator_correct

    def metrics(self, real_images, fake_images, real_labels, fake_labels, real_predicted, fake_predicted, generator_fake_predicted):
        # fid calculation
        with th.no_grad():
            self.discriminator.eval()
            fid = evaluation.calculate_fretchet(real_images, fake_images, self.discriminator)
            self.discriminator.train()

        real_correct = (real_labels == real_predicted.round()).sum().item()
        fake_correct = (fake_labels == fake_predicted.round()).sum().item()
        generator_correct = (real_labels == generator_fake_predicted.round()).sum().item()
        return fid, real_correct, fake_correct, generator_correct

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
        

    def save_model_metrics(self, epoch: int) -> None:
        self.make_epoch_directories(epoch)    
        metrics_path = f'{self.full_path}/{epoch}/metrics.csv'
        with open(metrics_path, 'w+') as f:
            writer = csv.writer(f)
            writer.writerow(self.model_metrics.keys())
            writer.writerows(zip(*self.model_metrics.values()))

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

    def save_image_grid(self, images, path: str, title: str) -> None:
        plt.figure(figsize=(8,8))
        plt.axis('off')
        plt.title(title)
        plt.imshow(np.transpose(vutils.make_grid(images.to(self.device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
        plt.savefig(path)
        plt.close()

    def make_epoch_directories(self, epoch: int) -> None:
        epoch_path = f'{self.full_path}/{epoch}'
        if not os.path.isdir(epoch_path):
            os.mkdir(epoch_path)
