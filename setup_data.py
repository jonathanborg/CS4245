import os
from PIL import Image
from tqdm import tqdm

directory = './data/faces/face'

sizes = set()
for filename in tqdm(os.listdir(directory)):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        image = Image.open(f)
        width, height = image.size
        sizes.add(f'{width}/{height}')

print(sizes)