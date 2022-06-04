import torch as th
import torchvision
from torch.utils.data import DataLoader

import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

"""
Network configurations
"""
# Network
NOISE_SIZE = 100
NOISE_TYPE = 'normal' 
CRITIC_FEATURE_MAP_DEPTH = 64               # in WGAN the Discriminator is called the Critic
GENERATOR_FEATURE_MAP_DEPTH = 64

# Training 
SAVE_CHECKPOINT_EVERY = 10 
SAVE_IMAGE_EVERY = 10
BATCH_SIZE = 64
EPOCHS = 1000
DISCRIMINATOR_LR = 1e-4
GENERATOR_LR = 1e-4 
TRUE_LABEL_VALUE = 1
FAKE_LABEL_VALUE = 0

# WGAN params
NUM_EPOCHS = 5
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.1

LAMBDA_GP = 10
VERSION = 0

"""
Model
"""
# DISCRIMINATOR
class CriticBlock(th.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, first: bool = False, last: bool = False) -> None:
        assert(not (first and last)) # block can't be both first and last
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
                th.nn.BatchNorm2d(out_channels),
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
        assert(not (first and last)) # block can't be both first and last
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
def train(generator, critic, generator_optimizer, critic_optimizer, device, dataloader, config):
    print("VERSION:", VERSION)
    for epoch in range(EPOCHS + 1):
        print('EPOCH: ', epoch)

        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            batch_size = real.size(0)
            
            # TRAIN DISCRIMINATOR (CRITIC) MORE. (5x according to paper)
            for _ in range(CRITIC_ITERATIONS):
                noise = th.randn(batch_size, NOISE_SIZE, 1, 1, device=device)
                fake = generator(noise)
                
                critic_fake = critic(fake).reshape(-1)
        
                critic_real = critic(real).reshape(-1)
                
                gp = gradient_penalty(critic, real, fake, device=device)
                
                # extra '-' because originally we want to maximize, so we minimize the negative.
                # LAMDA_GP * gp is the addition for WGAN-GP
                loss_critic = -(th.mean(critic_real) - th.mean(critic_fake)) + LAMBDA_GP * gp
                
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                critic_optimizer.step()
                
            
            # TRAIN GENERATOR 
            output = critic(fake).reshape(-1)
            loss_generator = -th.mean(output)
            generator.zero_grad() 
            loss_generator.backward()
            generator_optimizer.step()


        # SAVE MODEL AND IMAGES
        if epoch % SAVE_CHECKPOINT_EVERY == 0:
            print('-> Saving model checkpoint')
            save_model_checkpoint(epoch)
        
        if epoch % SAVE_IMAGE_EVERY == 0:
            print('-> Saving model images')
            save_model_image(epoch)