from os import path
import os
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

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
        if not path.isdir(self.results_path):
            os.mkdir(self.results_path)
        assert not os.isdir(self.full_path, f'Experiment with name {self.experiment_name} already exists in directory {self.results_path}')
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
        self.save_checkpoint_every = config['save_checkpoint_every']
        self.save_image_every = config['save_image_every']
        # constants
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.fixed_noise = th.randn(64, config['noise_size'], 1, 1, device=self.device)

    def train(self):
        for epoch in range(self.epochs):
            self.epoch()

    def epoch(self):
        for epoch, (real, _) in tqdm(enumerate(self.dataloader)):
            self.batch(real)
            if epoch % self.save_checkpoint_every and epoch > 0:
                self.save_model_checkpoint(epoch)
            if epoch % self.save_image_every and epoch > 0:
                self.save_model_image(epoch)

    def batch(self):
        pass

    def save_model_checkpoint(self, epoch: int) -> None:
        pass

    def save_model_image(self, epoch: int) -> None:
        pass