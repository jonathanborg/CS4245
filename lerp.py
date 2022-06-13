from fid import calculate_fid_given_paths, compute_statistics_of_path, InceptionV3, calculate_frechet_distance
import torch
import os
from torchvision.utils import save_image
import shutil
import matplotlib.pyplot as plt

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # config options
    model_type = 'wgan' # dcgan/wgan
    checkpoint_path = '/Users/fransdeboer/Projects/CS4245/results/results/wgan_augmented/120/checkpoint.th'
    # load either dcgan or wgan generator
    if model_type == 'dcgan':
        from models import Generator
    else:
        from models_wgan import Generator 
    # create constant noise
    torch.manual_seed(51)
    noise_start = torch.randn(1, 100, 1, 1, device=get_device()) - 0.25 
    torch.manual_seed(54)
    noise_end = torch.randn(1, 100, 1, 1, device=get_device()) + 0.25
    # create generator
    generator = Generator(100, 64).to(get_device())
    # load checkpoint into generator
    checkpoint = torch.load(checkpoint_path, map_location=get_device())
    generator.load_state_dict(checkpoint['generator_model_state_dict'])

    steps = 50
    step_tensor = torch.FloatTensor(noise_start.shape).fill_(1 / (steps - 1))
    for i in range(steps):
        weight = i * step_tensor
        lerped_noise = torch.lerp(noise_start, noise_end, weight)
        image = generator(lerped_noise)
        save_image(image, f'./lerped/{i}.png')

if __name__ == '__main__':
    main()
