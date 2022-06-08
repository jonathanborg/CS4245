import torch as th
import torchvision
from torch.utils.data import DataLoader

import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

from tqdm import tqdm

from experiments import evaluation
import csv

"""
Model
"""


# DISCRIMINATOR
class CriticBlock(th.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, first: bool = False, last: bool = False) -> None:
        assert (not (first and last))  # block can't be both first and last
        super().__init__()
        if first:
            self.main = th.nn.Sequential(
                th.nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                th.nn.LeakyReLU(0.2, inplace=True),
            )

        elif last:
            self.main = th.nn.Sequential(
                th.nn.Conv2d(in_channels, out_channels, 3, 1, 0, bias=False),
                # No Sigmoid activation in WGAN in last layer
            )

        else:
            self.main = th.nn.Sequential(
                th.nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False),
                th.nn.InstanceNorm2d(out_channels),
                th.nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.main(x)


class Critic(th.nn.Module):
    def __init__(self, feature_map_depth: int) -> None:
        super().__init__()
        self.main = th.nn.Sequential(
            CriticBlock(3, feature_map_depth, first=True),
            CriticBlock(feature_map_depth, feature_map_depth * 2),
            CriticBlock(feature_map_depth * 2, feature_map_depth * 4),
            CriticBlock(feature_map_depth * 4, feature_map_depth * 8),
            CriticBlock(feature_map_depth * 8, feature_map_depth * 8),
            CriticBlock(feature_map_depth * 8, 1, last=True)
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.main(x)
        return x


# GENERATOR
class GeneratorBlock(th.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, first: bool = False, last: bool = False) -> None:
        assert (not (first and last))  # block can't be both first and last
        super().__init__()
        if first:
            self.main = th.nn.Sequential(
                th.nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 0, bias=False),
                th.nn.BatchNorm2d(out_channels),
                th.nn.ReLU(True)
            )
        elif last:
            self.main = th.nn.Sequential(
                th.nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                th.nn.Tanh()
            )
        else:
            self.main = th.nn.Sequential(
                th.nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
                th.nn.BatchNorm2d(out_channels),
                th.nn.ReLU(True)
            )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.main(x)


class Generator(th.nn.Module):
    def __init__(self, noise_size: int, feature_map_depth: int) -> None:
        super().__init__()
        # first layer, no stride. Upsample from 1x1 to 4x4
        self.main = th.nn.Sequential(
            GeneratorBlock(noise_size, feature_map_depth * 8, first=True),
            GeneratorBlock(feature_map_depth * 8, feature_map_depth * 8),
            GeneratorBlock(feature_map_depth * 8, feature_map_depth * 4),
            GeneratorBlock(feature_map_depth * 4, feature_map_depth * 2),
            GeneratorBlock(feature_map_depth * 2, feature_map_depth * 1),
            GeneratorBlock(feature_map_depth * 1, 3, last=True),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.main(x)
        return x


# GRADIENT PENALTY (WGAN-GP)
def gradient_penalty(critic, real, fake, device="cpu"):
    # Create interpolated image
    BATCH_SIZE, C, H, W = real.shape
    alpha = th.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)

    # Critic score of interpolated image
    mixed_scores = critic(interpolated_images)

    # Take gradients of scores with respect to the interpolated images
    gradient = th.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=th.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = th.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


# MAIN TRAINING LOOP
class Training:
    def __init__(self, generator, critic, generator_optimizer, critic_optimizer, device, dataloader, config):
        self.results_path = config['local_results_directory']
        self.experiment_name = config['experiment_name']
        self.full_path = f'{self.results_path}/{self.experiment_name}'
        if not os.path.isdir(self.full_path):
            os.mkdir(self.full_path)
        self.evaluation = config['evaluation']

        # store models, optimizers, criterion, and data
        self.generator = generator
        self.critic = critic
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer
        self.dataloader = dataloader

        self.evaluation = config['evaluation']
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
        self.model_name = config['model_name']
        self.device = device
        self.fixed_noise = th.randn(64, self.noise_size, 1, 1, device=self.device)
        self.critic_iterations = config['critic_iterations']
        self.lambda_gp = config['lambda_gp']
        self.wgan_gp_lr = config['wgan_gp_lr']
        self.wgan_gp_betas = config['wgan_gp_betas']

        self.datasize = len(self.dataloader.dataset)
        self.model_metrics = {
            'epochs': [],
            'fid': [],
            'loss_g': [],
            'loss_d': [],
            'activations_g': [],
            'activations_d_real': [],
            'activations_d_fake': []
        }

    def train(self, current_epoch):
        evaluation_outcomes = []
        for epoch in range(current_epoch, self.epochs + 1):
            print('EPOCH: ', epoch)

            total_fid = 0
            total_critic_error = 0
            total_generator_error = 0

            total_real_activations = 0
            total_fake_activations = 0
            total_generator_activations = 0
            for (real, _) in tqdm(self.dataloader, leave=False):
                # for batch_idx, (real, _) in enumerate(self.dataloader):
                real = real.to(self.device)
                batch_size = real.size(0)

                # TRAIN DISCRIMINATOR (CRITIC) MORE. (5x according to paper)
                for _ in range(self.critic_iterations):
                    noise = th.randn(batch_size, self.noise_size, 1, 1, device=self.device)
                    fake = self.generator(noise)
                    critic_fake = self.critic(fake).reshape(-1)
                    critic_real = self.critic(real).reshape(-1)
                    gp = gradient_penalty(self.critic, real, fake, device=self.device)

                    # extra '-' because originally we want to maximize, so we minimize the negative.
                    # LAMDA_GP * gp is the addition for WGAN-GP
                    loss_critic = -(th.mean(critic_real) - th.mean(critic_fake)) + self.lambda_gp * gp

                    self.critic.zero_grad()
                    loss_critic.backward(retain_graph=True)
                    self.critic_optimizer.step()

                # TRAIN GENERATOR
                output = self.critic(fake).reshape(-1)
                loss_generator = -th.mean(output)
                self.generator.zero_grad()
                loss_generator.backward()
                self.generator_optimizer.step()

                if self.evaluation:
                    batch_size = real.size(0)
                    real_labels = th.full((batch_size,), self.true_label_value, dtype=th.float, device=self.device)
                    fake_labels = th.full((batch_size,), self.fake_label_value, dtype=th.float, device=self.device)
                    # fid, real_correct, fake_correct, generator_correct = self.metrics(real, fake, real_labels,
                    #                                                                   fake_labels, critic_real,
                    #                                                                   critic_fake, output)

                    total_fid += 0
                    total_real_activations += critic_real.sum().item()
                    total_fake_activations += critic_fake.sum().item()
                    total_generator_activations += output.sum().item()
                    total_critic_error += loss_critic.item()
                    total_generator_error += loss_generator.item()

                    evaluation_outcomes.append(
                        [total_fid, total_critic_error, total_generator_error, total_real_activations,
                         total_fake_activations, total_generator_activations])

            # metric calculation
            fid = total_fid / self.datasize
            critic_error = total_critic_error / len(self.dataloader)
            generator_error = total_generator_error / len(self.dataloader)

            fake_activations = total_fake_activations / self.datasize
            real_activations = total_real_activations / self.datasize
            activations_g = total_generator_activations / self.datasize

            loss_string = f'Loss_G: {generator_error:.4f}\tLoss_D: {critic_error:.4f}'
            accuracy_string = f'Activations_G: {activations_g:.4f}\tReal_Activations_D: {real_activations:.4f}\tFake_Activations_D: {fake_activations:.4f}'
            print(f'{epoch + 1}/{self.epochs}: FID: {fid}\t{loss_string}\t{accuracy_string}')

            self.model_metrics['epochs'].append(epoch)
            self.model_metrics['fid'].append(fid)
            self.model_metrics['loss_g'].append(generator_error)
            self.model_metrics['loss_d'].append(critic_error)
            self.model_metrics['activations_g'].append(activations_g)
            self.model_metrics['activations_d_real'].append(real_activations)
            self.model_metrics['activations_d_fake'].append(fake_activations)

            # SAVE MODEL AND IMAGES
            if epoch % self.save_checkpoint_every == 0:
                print('-> Saving model checkpoint')
                self.save_model_checkpoint(epoch)

            if epoch % self.save_image_every == 0:
                print('-> Saving model images')
                self.save_model_image(epoch)

            if epoch % self.save_metrics_every == 0:
                print('-> Saving metrics')
                self.save_model_metrics(epoch, self.model_metrics)

        return evaluation_outcomes

    def metrics(self, real_images, fake_images, real_labels, fake_labels, real_predicted, fake_predicted,
                generator_fake_predicted):
        # fid calculation
        # with th.no_grad():
        # self.critic.eval()
        fid = 0  # evaluation.calculate_fretchet(real_images, fake_images, self.critic)
        # self.critic.train()

        real_correct = real_predicted.sum().item()
        fake_correct = fake_predicted.sum().item()
        generator_correct = generator_fake_predicted.sum().item()
        return fid, real_correct, fake_correct, generator_correct

    def save_model_checkpoint(self, epoch: int) -> None:
        self.make_epoch_directories(epoch)
        checkpoint_path = f'{self.full_path}/{epoch}'
        th.save({
            'epoch': epoch,
            'generator_model_state_dict': self.generator.state_dict(),
            'discriminator_model_state_dict': self.critic.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.critic_optimizer.state_dict(),
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

    def load_model_checkpoint(self, path: str) -> int:
        ordered_saves = os.listdir(path)
        ordered_saves.sort(reverse=True)
        latest_save = ordered_saves[0]
        checkpoint_path = f'{path}/{latest_save}/checkpoint.th'
        checkpoint = th.load(checkpoint_path)

        self.generator.load_state_dict(checkpoint['generator_model_state_dict'])
        self.critic.load_state_dict(checkpoint['discriminator_model_state_dict'])
        self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        return checkpoint['epoch']
