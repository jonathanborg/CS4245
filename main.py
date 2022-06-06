import os
import shutil

model_to_train = 'dcgan'
# model_to_train = 'wgan-gp'

config = {
    # environment
    'environment': 'local',
    'local_results_directory': './results',
    'experiment_name': 'v1',
    'data_directory': './data/faces',
    'evaluation': True,
    'num_workers': 8,

    # network
    'noise_size': 100,
    'noise_type': 'normal', # uniform / normal
    'discriminator_feature_map_depth': 64,
    'generator_feature_map_depth': 64,

    # training
    'save_checkpoint_every': 3,
    'save_image_every': 3,
    'save_metrics_every': 3,
    'batch_size': 64,
    'epochs': 101,
    'discriminator_lr': 0.002,
    'discriminator_betas': (0.5, 0.999),
    'generator_lr': 0.002,
    'generator_betas': (0.5, 0.999),
    'true_label_value': 1,
    'fake_label_value': 0,

    # model
    'model_name': model_to_train,

    # model specific settings
    # wgan settings
    'weight_clip': 0.1,

    # wgan-gp settings
    'critic_iterations': 5,
    'lambda_gp': 10,
    'wgan_gp_lr': 1e-4,
    'wgan_gp_betas': (0.0, 0.9)

}

# create paths
# shutil.rmtree(config['local_results_directory'])
if not os.path.isdir(config['local_results_directory']):
    os.mkdir(config['local_results_directory'])

import torch as th
import torchvision
from torch.utils.data import DataLoader

from models import Generator, Discriminator
import models_wgan.wgan_gp as wgan_gp
from utils import weights_init
from experiments import Experiment

# create device
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

# create dataset
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
dataset = torchvision.datasets.ImageFolder(config['data_directory'], transform=transform)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

if config['model_name'] == 'dcgan':
    # create networks
    generator = Generator(config['noise_size'],config['generator_feature_map_depth']).to(device)
    discriminator = Discriminator(config['discriminator_feature_map_depth']).to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # create optimizers
    discriminator_optimizer = th.optim.Adam(discriminator.parameters(), lr=config['discriminator_lr'], betas=config['discriminator_betas'])
    generator_optimizer = th.optim.Adam(generator.parameters(), lr=config['generator_lr'], betas=config['generator_betas'])

    # create loss
    criterion = th.nn.BCELoss(reduction='sum')
    # create experiment
    experiment = Experiment(config, generator, discriminator, generator_optimizer, discriminator_optimizer, criterion, dataloader)
    experiment.train()


# elif config['model_name'] == 'dcgan-data-aug':
#
# elif config['model_name'] == 'wgan':
#
elif config['model_name'] == 'wgan-gp':
    # create networks
    generator = wgan_gp.Generator(config["noise_size"], config["generator_feature_map_depth"]).to(device)
    critic = wgan_gp.Critic(config["discriminator_feature_map_depth"]).to(device)
    generator.apply(weights_init)
    critic.apply(weights_init)

    # create optimizers
    # Optimizer (WGAN uses RMSprop, WGAN-GP uses Adam)
    critic_optimizer = th.optim.Adam(critic.parameters(), lr=config["wgan_gp_lr"], betas=config["wgan_gp_betas"])
    generator_optimizer = th.optim.Adam(generator.parameters(), lr=config["wgan_gp_lr"], betas=config["wgan_gp_betas"])

    generator.train()
    critic.train()
    criterion = None

    experiment = wgan_gp.Training(generator, critic, generator_optimizer, critic_optimizer, device, dataloader, config)
    experiment.train()

# elif config['model_name'] == 'wgan-gp-data-aug':
