from fid import calculate_fid_given_paths
import torch
import os
from torchvision.utils import save_image
import shutil

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # config options
    model_type = 'wgan' # dcgan/wgan
    num_generated = 100
    data_path = '/Users/fransdeboer/Projects/CS4245/data/faces_reduced/face'
    results_path = '/Users/fransdeboer/Downloads/wgan_augmented'
    # load either dcgan or wgan generator
    if model_type == 'dcgan':
        from models import Generator
    else:
        from models_wgan import Generator 
    # create constant noise
    noise = torch.randn(num_generated, 100, 1, 1, device=get_device())   
    # create generator
    generator = Generator(100, 64)
    # load all checkpoint directories
    fid_values = []
    checkpoint_directories = sorted(os.listdir(results_path))
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
        fid_value = calculate_fid_given_paths([data_path,'./fid_images'],
                                            500,
                                            get_device(),
                                            2048)
        print(fid_value)
        fid_values.append(fid_value)
        shutil.rmtree('./fid_images')
    print(fid_values)

if __name__ == '__main__':
    main()