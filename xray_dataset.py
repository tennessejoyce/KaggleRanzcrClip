import torch
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from torchvision.transforms import Normalize

def random_mask(length, num_false):
    """
    Returns a boolean array of specified length, with specified number of false elements.
    """
    mask = np.ones(length)
    mask[:num_false]=0
    np.random.shuffle(mask)
    return mask.astype(bool)


def read_image(image_file, resolution):
    """
    Reads in the xray image from a file, resizes, and returns an nparray.
    """
    try:
        image = Image.open(image_file)
        image = image.resize((resolution, resolution), Image.BILINEAR)
        return np.array(image)
    except:
        print(f'Failed to load {image_file}')
        return np.zeros((resolution, resolution), dtype=np.uint8)


def read_all_images(stage, instance_ids, resolution):
    save_filename = f'{stage}_{resolution}.npy'
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


def load_dataset(stage='train', resolution=224, val_type=None, val_fraction=0.2, val_num_folds=5, random_state=0,
                 drop_fraction=0):
    """
    Loads both the images and the binary targets into numpy arrays.
    Also splits training data into traning and validation, keeping rows from the same
    patient together to avoid leakage. Validation fraction specifies the fraction of
    patients, chosen at random, who are assigned to the validation set.
    """
    # Read in the tabular data, with the instance_id as the index column.
    df = pd.read_csv(f'data/{stage}.csv', index_col=0)
    # Separate out the PatientID column, leaving only the binary targets.
    patient_id = df.pop('PatientID')
    # Loop over rows, building up the image and target data into lists.
    target_data = df.values
    image_data = read_all_images(stage='train', instance_ids=df.index, resolution=resolution)


    # Drop some fraction of the rows for testing purposes.
    mask = random_mask(df.shape[0], int(df.shape[0]*drop_fraction))
    patient_id = patient_id.iloc[mask]
    target_data = target_data[mask]
    image_data = image_data[mask]

    if val_type is None:
        return XRayDataset(image_data, target_data)
    elif val_type == 'split':
        unique_ids = patient_id.unique()
        val_size = int(val_fraction * len(unique_ids))
        rng = np.random.default_rng(random_state)
        validation_patients = rng.choice(unique_ids, val_size)
        validation_idx = patient_id.isin(validation_patients)
        train_idx = ~validation_idx
        train_dataset = XRayDataset(image_data[train_idx], target_data[train_idx])
        val_dataset = XRayDataset(image_data[validation_idx], target_data[validation_idx])
        return train_dataset, val_dataset
    else:
        print(f'{val_type} cross validation has not been implemented yet.')
        return XRayDataset(image_data, target_data)


class XRayDataset(Dataset):
    """
    Pytorch dataset that stores both the image data and binary target data
    in memory as numpy arrays.
    """

    def __init__(self, image_data, target_data, grayscale=False):
        assert image_data.shape[0] == target_data.shape[0]
        self.target_data = torch.from_numpy(target_data)
        # Add an extra index for channels (gray scale).
        self.image_data = torch.from_numpy(image_data[:, None, :, :])
        class_proportions = torch.from_numpy(np.mean(target_data, axis=0))
        self.weights = 0.5 * (self.target_data/class_proportions + (1-self.target_data)/(1-class_proportions))
        # Standard imagenet normalizer
        if grayscale:
            self.normalzer = lambda x: x
            self.channels = 1
        else:
            self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.channels = 3

    def __getitem__(self, key):
        image_batch = self.to_normalized_rgb(self.image_data[key].float()).cuda()
        target_batch = self.target_data[key].float().cuda()
        weights_batch = self.weights[key].float().cuda()
        return image_batch, target_batch, weights_batch

    def __len__(self):
        return self.image_data.shape[0]

    def to_normalized_rgb(self, grayscale_image):
        """
        Converts a grayscale image (as a 2d nparray) into RGB, and applies the Imagenet normalization.
        """
        rgb_image = grayscale_image.repeat_interleave(self.channels, 0)
        return self.normalizer(rgb_image)

