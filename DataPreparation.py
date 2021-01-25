import torch
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.transforms import Normalize

def random_mask(length, num_false):
    """
    Return a random boolean mask, as an nparray.

    Keyword arguments:
    length -- the length of the mask
    num_false -- the number of elements which are false (should be <= length)
    """
    mask = np.ones(length)
    mask[:num_false]=0
    np.random.shuffle(mask)
    return mask.astype(bool)


def read_image(image_file, resolution):
    """
    Read an xray image from a file, resize, and return an nparray.

    Keyword arguments:
    image_file -- the name of the image file to be read
    resolution -- the resolution of the image (e.g. 256)
    """
    try:
        image = Image.open(image_file)
        image = image.resize((resolution, resolution), Image.BILINEAR)
        return np.array(image)
    except:
        print(f'Failed to load {image_file}')
        return np.zeros((resolution, resolution), dtype=np.uint8)


def read_all_images(stage, instance_ids, resolution):
    save_filename = f'data/intermediate_data/{stage}_{resolution}.npy'
    try:
        return np.load(save_filename)
    except:
        print('Loading images from jpg...')
    image_data = []
    for index in tqdm(instance_ids):
        image_data.append(read_image(f'data/{stage}/{index}.jpg', resolution))
    image_data = np.stack(image_data)
    np.save(save_filename, image_data)
    return image_data

class ImageLoader:
    def __init__(self, stage):
