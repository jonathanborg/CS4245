"""
WGAN implementation of the model.
"""

from xml.dom.expatbuilder import parseString
import torch as th
import torchvision
from torch.utils.data import DataLoader

from models_wgan import Generator, Discriminator
from utils import weights_init
# from experiments import Experiment
import os
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

config = {
    # environment
    'environment': 'local', # local / kaggle (TODO: Implement for Kaggle)
    'local_results_directory': './results',
    'experiment_name': 'v2.6.3',
    'data_directory': './data/faces',

    # network
    'noise_size': 100,
    'noise_type': 'normal', # uniform / normal
    'discriminator_feature_map_depth': 64,
    'generator_feature_map_depth': 64,

    # training
    'save_checkpoint_every': 5,
    'save_image_every': 5,
    'batch_size': 64,
    'epochs': 1000,
    'discriminator_lr': 5e-5,
    'discriminator_betas': (0.5, 0.999),
    'generator_lr': 5e-5,
    'generator_betas': (0.5, 0.999),
    'true_label_value': 1,
    'fake_label_value': 0,

    # WGAN params
    'num_epochs': 5,
    'critic_iterations': 5,
    'weight_clip': 0.1
}

# create device
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print("device:", device)
print("is cuda available: ", th.cuda.is_available())

# create dataset
transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
dataset = torchvision.datasets.ImageFolder(config['data_directory'], transform=transform)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

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
discriminator_optimizer = th.optim.RMSprop(discriminator.parameters(), lr=config['discriminator_lr'])
generator_optimizer = th.optim.RMSprop(generator.parameters(), lr=config['generator_lr'])


"""
UTILITY FUNCTIONS
"""
results_path = config['local_results_directory']
experiment_name = config['experiment_name']
FULL_PATH = f'{results_path}/{experiment_name}'
fixed_noise = th.randn(64, config['noise_size'], 1, 1, device=device)

def save_model_checkpoint(epoch: int) -> None:
    make_epoch_directories(epoch)
    checkpoint_path = f'{FULL_PATH}/{epoch}'
    th.save({
        'epoch': epoch,
        'generator_model_state_dict': generator.state_dict(),
        'discriminator_model_state_dict': discriminator.state_dict(),
        'generator_optimizer_state_dict': generator_optimizer.state_dict(),
        'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
    }, f'{checkpoint_path}/checkpoint.th')


def make_epoch_directories(epoch: int) -> None:
    epoch_path = f'{FULL_PATH}/{epoch}'
    if not os.path.isdir(epoch_path):
        os.mkdir(epoch_path)


def save_model_image(epoch: int) -> None:
    make_epoch_directories(epoch)
    image_path = f'{FULL_PATH}/{epoch}/images'
    if not os.path.isdir(image_path):
        os.mkdir(image_path)
    random_noise = th.randn(64, config['noise_size'], 1, 1, device=device)
    fixed_fakes = generator(fixed_noise).detach().cpu()
    random_fakes = generator(random_noise).detach().cpu()
    save_image_grid(fixed_fakes, f'{image_path}/fixed.png', 'Fixed Noise')
    save_image_grid(random_fakes, f'{image_path}/random.png', 'Random Noise')


def save_image_grid(images, path: str, title: str) -> None:
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title(title)
    plt.imshow(np.transpose(vutils.make_grid(images.to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig(path)
    plt.close()



# TRAINING LOOP

for epoch in range(config['epochs']):
    print('EPOCH: ', epoch)
    for batch_idx, (real, _) in enumerate(dataloader):
        print('\t batch:', batch_idx)
        real = real.to(device)
        batch_size = real.size(0)
        
        # TRAIN DISCRIMINATOR (CRITIC) MORE. (5x according to paper)
        for _ in range(config['critic_iterations']):
            noise = th.randn(batch_size, config['noise_size'], 1, 1, device=device)
            global fake 
            fake = generator(noise)
            
            discriminator_fake = discriminator(fake).reshape(-1)
    
            discriminator_real = discriminator(real).reshape(-1)

            loss_discriminator = -(th.mean(discriminator_fake) - th.mean(discriminator_real))

            discriminator.zero_grad()
            loss_discriminator.backward(retain_graph=True)
            discriminator_optimizer.step()

            for p in discriminator.parameters():
                p.data.clamp_(-config['weight_clip'], config['weight_clip'])

        # TRAIN GENERATOR 
        output = discriminator(fake).reshape(-1)
        loss_generator = -th.mean(output)
        generator.zero_grad() 
        loss_generator.backward()
        generator_optimizer.step()

    # SAVE IMAGES
    if epoch % config['save_checkpoint_every'] == 0:
        print('-> Saving model checkpoint')
        save_model_checkpoint(epoch)

    if epoch % config['save_image_every'] == 0:
        print('-> Saving model images')
        save_model_image(epoch)
