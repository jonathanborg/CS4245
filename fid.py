from fid import calculate_fid_given_paths, compute_statistics_of_path, InceptionV3, calculate_frechet_distance
import torch
import os
from torchvision.utils import save_image
import shutil

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # config options
    model_type = 'wgan' # dcgan/wgan
    num_generated = 1000
    data_path = '/home/frans/Projects/CS4245/data/faces/face'
    results_path = '/home/frans/Projects/CS4245/results/wgan_augmented'
    # load either dcgan or wgan generator
    if model_type == 'dcgan':
        from models import Generator
    else:
        from models_wgan import Generator 
    # create constant noise
    noise = torch.randn(num_generated, 100, 1, 1, device=get_device())   
    # create generator
    generator = Generator(100, 64).to(get_device())
    # load all checkpoint directories
    fid_values = []
    checkpoint_directories = sorted(os.listdir(results_path))
    # precompute data m1, s1
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]

    model = InceptionV3([block_idx]).to(get_device())
    m1, s1 = compute_statistics_of_path(data_path, model, 32,
                                        2048, get_device(), 8)

    for checkpoint_index in checkpoint_directories:
        print(f'Evaluating {checkpoint_index}')
        # load checkpoint file
        checkpoint_path = os.path.join(results_path, checkpoint_index, 'checkpoint.th')
        checkpoint = torch.load(checkpoint_path, map_location=get_device())
        # load state dict into generator
        generator.load_state_dict(checkpoint['generator_model_state_dict'])
        # create images
        images = generator(noise)
        # create image directory
        os.mkdir('./fid_images')
        # save each generated image
        for i,image in enumerate(images):
            save_image(image, f'./fid_images/image_{i}.png')
        # calculate fid value
        m2, s2 = compute_statistics_of_path('./fid_images', model, 32,
                                            2048, get_device(), 8)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        print(fid_value)
        fid_values.append(fid_value)
        shutil.rmtree('./fid_images')
        torch.cuda.empty_cache()
    print(fid_values)

def graphs():
    pass

if __name__ == '__main__':
    graphs()
