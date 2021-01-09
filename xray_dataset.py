import torch
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


def read_image(image_file, resolution):
    """Reads in the xray image from a file, resizes, and returns an nparray."""
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


def load_dataset(stage='train', resolution=224, val_fraction=0, random_state=0):
    """Loads both the images and the binary targets into a Pytorch Dataset object.
    Also splits training data into traning and validation, keeping rows from the same
    patient together to avoid leakage. Validation fraction specifies the fraction of
    patients, chosen at random, who are assigned to the validation set."""
    # Read in the tabular data, with the instance_id as the index column.
    df = pd.read_csv(f'data/{stage}.csv', index_col=0)
    # Separate out the PatientID column, leaving only the binary targets.
    patient_id = df.pop('PatientID')
    # Loop over rows, building up the image and target data into lists.
    target_data = df.values
    image_data = read_all_images(stage='train', instance_ids=df.index, resolution=resolution)
    # If we're not doing validation, just return one dataset.
    if val_fraction == 0:
        return XRayDataset(image_data, target_data)
    # Otherwise split patients into train and validation, and return two datasets.
    else:
        unique_ids = patient_id.unique()
        val_size = int(val_fraction * len(unique_ids))
        rng = np.random.default_rng(random_state)
        validation_patients = rng.choice(unique_ids, val_size)
        validation_idx = patient_id.isin(validation_patients)
        train_idx = ~validation_idx
        train_dataset = XRayDataset(image_data[train_idx], target_data[train_idx])
        validation_dataset = XRayDataset(image_data[validation_idx], target_data[validation_idx])
        return train_dataset, validation_dataset


class XRayDataset (Dataset):
    """Pytorch dataset that stores both the image data and binary target data
    in memory as numpy arrays."""
    def __init__(self, image_data, target_data):
        assert image_data.shape[0] == target_data.shape[0]
        self.target_data = torch.from_numpy(target_data)
        # Add an extra index for channels (gray scale).
        self.image_data = torch.from_numpy(image_data[:, None, :, :])

    def __getitem__(self, key):
        image_batch = self.image_data[key]
        return image_batch, self.target_data[key]

    def __len__(self):
        return self.image_data.shape[0]

    def ready(self, device):
        self.target_data = self.target_data.float().to(device)
        self.image_data = self.image_data.float().to(device)
