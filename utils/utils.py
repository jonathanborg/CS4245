import torch as th
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np


def weights_init(model):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        th.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        th.nn.init.normal_(m.weight.data, 1.0, 0.02)
        th.nn.init.constant_(m.bias.data, 0)
 
def save_images(images, device, name, path='./generator_results'):
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.title('Images')
    plt.imshow(np.transpose(vutils.make_grid(images[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig(f'{path}/{name}')
    plt.close()