import os
import shutil

# import utils.utils
# utils.utils.resize_images('C:\\Development\\CS4245\\data\\cartoonset100k\\', 'C:\\Development\\CS4245\\data\\cartoon_faces\\all_faces\\')
# import utils.utils
# utils.utils.add_label_to_image('C:\\Development\\CS4245\\data\\pre-trained models\\dcgan_dataaug\\', 'C:\\Development\\CS4245\\data\\gif_gen\\', "Epoch")


# model_to_train = 'dcgan'
model_to_train = 'wgan-gp'

config = {
    # environment
    'environment': 'local',
    'local_results_directory': './results',
    'experiment_name': 'save_func_wgan',
    'data_directory': './data/faces_reduced',
    'prior_training': './kernel_output/generator_results',
    'evaluation': True,
    'num_workers': 0,

    # network
    'noise_size': 100,
    'noise_type': 'normal',  # uniform / normal
    'discriminator_feature_map_depth': 64,
    'generator_feature_map_depth': 64,

    # training
    'save_checkpoint_every': 10,
    'save_image_every': 10,
    'save_metrics_every': 10,
    'batch_size': 64,
    'epochs': 100001,
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
# Load prior training and continue training on latest checkpoint
load_model = os.path.isdir(config['prior_training']) and config['experiment_name'] in os.listdir(config['prior_training'])

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
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.RandomHorizontalFlip()])
dataset = torchvision.datasets.ImageFolder(config['data_directory'], transform=transform)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

if config['model_name'] == 'dcgan':
    # create networks
    generator = Generator(config['noise_size'], config['generator_feature_map_depth']).to(device)
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

current_epoch = 0
if load_model:
    current_epoch = experiment.load_model_checkpoint(os.path.join(config['prior_training'], config['experiment_name']))
experiment.train(current_epoch)
# elif config['model_name'] == 'wgan-gp-data-aug':
