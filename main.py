import torch as th
import torchvision
from torch.utils.data import DataLoader

from models import Generator, Discriminator
from utils import weights_init
from experiments import Experiment

config = {
    # environment
    'environment': 'local', # local / kaggle (TODO: Implement for Kaggle)
    'local_results_directory': './results',
    'experiment_name': 'v1',
    'data_directory': './data/faces',
    'evaluation': False,
    'num_workers': 8,
    # network
    'noise_size': 100,
    'discriminator_feature_map_depth': 64,
    'generator_feature_map_depth': 64,
    # training
    'save_checkpoint_every': 10,
    'save_image_every': 10,
    'batch_size': 128,
    'epochs': 1000,
    'discriminator_lr': 0.002,
    'discriminator_betas': (0.5, 0.999),
    'generator_lr': 0.002,
    'generator_betas': (0.5, 0.999),
    'true_label_value': 1,
    'fake_label_value': 0,
}

# create device
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
# create dataset
transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
dataset = torchvision.datasets.ImageFolder(config['data_directory'], transform=transform)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
# create networks
generator = Generator(
    config['noise_size'],
    config['generator_feature_map_depth']
).to(device)
generator.apply(weights_init)
discriminator = Discriminator(
    config['discriminator_feature_map_depth']
).to(device)
discriminator.apply(weights_init)
# create optimizers
discriminator_optimizer = th.optim.Adam(discriminator.parameters(), lr=config['discriminator_lr'], betas=config['discriminator_betas'])
generator_optimizer = th.optim.Adam(generator.parameters(), lr=config['generator_lr'], betas=config['generator_betas'])
# create loss
criterion = th.nn.BCELoss()
# create experiment
experiment = Experiment(config, 
                        generator, 
                        discriminator, 
                        generator_optimizer,
                        discriminator_optimizer, 
                        criterion, 
                        dataloader)
experiment.train()