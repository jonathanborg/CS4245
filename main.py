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
    'experiment_name': '',
    'data_directory': './data/faces',
    # network
    'noise_size': 100,
    'discriminator_feature_map_depth': 64,
    'generator_feature_map_depth': 64,
    # training
    'save_checkpoint_every': 1,
    'save_image_every': 1,
    'batch_size': 128,
    'epochs': 1000,
    'discriminator_lr': 0.002,
    'discriminator_betas': (0.5, 0.999),
    'generator_lr': 0.002,
    'generator_betas': (0.5, 0.999),
}

# create device
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
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

# fixed_noise = th.randn(64, config['noise_size'], 1, 1, device=device)
# real_label = 1
# fake_label = 0

# for epoch in range(config['epochs']):
#     for (real, _) in tqdm(dataloader):
#         netD.zero_grad()
#         real = real.to(device)
#         b_size = real.size(0)
#         label = th.full((b_size, ), real_label, dtype=th.float, device=device)
#         output = netD(real).view(-1)
#         errD_real = criterion(output, label)
#         errD_real.backward()
#         D_x = output.mean().item()

#         noise = th.randn(b_size, config['noise_size'], 1, 1, device=device)
#         fake = netG(noise)
#         label.fill_(fake_label)
#         output = netD(fake.detach()).view(-1)
#         errD_fake = criterion(output, label)
#         errD_fake.backward()
#         optimizerD.step()


#         netG.zero_grad()
#         label.fill_(real_label)
#         output = netD(fake).view(-1)
#         errG = criterion(output, label)
#         errG.backward()
#         optimizerG.step()

#     with th.no_grad():
#         fake = netG(fixed_noise).detach().cpu()
#         plt.figure(figsize=(8,8))
#         plt.axis('off')
#         plt.title('Training images')
#         plt.imshow(np.transpose(vutils.make_grid(fake.to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
#         plt.show()
#         plt.savefig(f'./results/generated-{epoch}.png')
#         plt.close()