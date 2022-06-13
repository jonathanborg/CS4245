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
    print(m1, s1)

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
    # wgan data augmentation
    xs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    wgan_augmented = [368.6099345285654, 188.3532296306024, 170.79805863822912, 162.74850101333993, 145.2728353697495, 140.37733626842157, 140.9507854985846, 131.81067818681353, 133.38667535752313, 133.4297511465794, 131.8392472853391] # 131.42615702635283, 128.7624647286123]
    dcgan = [375.7762989142211, 305.8697058792776, 194.6081034839227, 169.37506422082535, 144.40715584978398, 142.4899873974968, 141.6846938467572, 406.6730606052551, 406.6731220816454, 406.6730964668362, 406.6729799928464]
    dcgan_augmented = [383.81914231043345, 308.55042001977665, 238.62345759284807, 184.91935426009096, 160.57244763658912, 135.51404418370788, 125.85816735232478, 160.79977303128283, 230.8415781770979, 281.4676042903392, 291.6101831497997]
    wgan = [205.79575045430678, 176.7015760836012, 157.08438287578983, 164.54241628456506, 150.15396916939517, 151.99910278472757, 155.25245682988066]
    plt.plot(xs, dcgan, label='dcgan')
    plt.plot(xs, dcgan_augmented, label='dcgan w/ augmentation')
    plt.plot([0, 10, 20, 30, 40, 50, 60], wgan, label='wgan')
    plt.plot(xs, wgan_augmented, label='wgan w/ augmentation')

    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('FID')
    plt.title('FID score of GANs (lower is better)')
    plt.show()

if __name__ == '__main__':
    graphs()
