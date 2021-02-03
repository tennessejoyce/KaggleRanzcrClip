import numpy as np
import pandas as pd
import torch
import torchvision.models
from torchvision.transforms import Normalize
from sklearn.decomposition import IncrementalPCA
from copy import copy
from PIL import Image
from tqdm import tqdm


def batch_generator(generator, batch_size):
    """Collects a one-at-a-time datastream into batches."""
    batch = []
    for i, item in enumerate(generator):
        batch.append(item)
        if (i + 1) % batch_size == 0:
            yield np.stack(batch)
            batch = []
    # Return any remaining items in a partial batch.
    if len(batch) > 0:
        yield np.stack(batch)


def load_cnn_headless(architecture):
    """Load a convolutional neural network of specified architecture into self['cnn']."""
    assert hasattr(torchvision.models, architecture), f'CNN architecture not found {architecture}'
    # Load the CNN with pretrained weights.
    cnn = getattr(torchvision.models, architecture)(pretrained=True)
    # Replace the last fully connected layer with the identity operation.
    cnn.fc = torch.nn.Identity()
    cnn.float().cuda()
    cnn.eval()
    return cnn


class FeatureExtraction:
    def __init__(self, cnn_architecture='resnet18', data_dir='data',
                 resolution=224, batch_size=256, random_state=0):
        # Directory containing all the data.
        self.data_dir = data_dir
        self.resolution = resolution
        self.batch_size = batch_size
        self.random_state = random_state

        # Load in the tabular data.
        train_df = pd.read_csv(f'{self.data_dir}/train.csv', index_col=0)
        test_df = pd.read_csv(f'{self.data_dir}/sample_submission.csv', index_col=0)
        # Add a column for the stage (train/test)
        train_df['stage'] = 'train'
        test_df['stage'] = 'test'
        # Combine them into a single dataframe.
        self.df = pd.concat([train_df, test_df])
        # Create a new, combined index for both train and test sets.
        self.df = self.df.reset_index()
        # Shuffle the rows, in case that affects PCA.
        self.df = self.df.sample(frac=1, random_state=self.random_state)

        # Instantiate models for CNN and PCA
        self.cnn = load_cnn_headless(cnn_architecture)
        self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.pca = IncrementalPCA()

    def image_generator(self, verbose=False):
        """A generator that loads all the images one at a time."""
        for _, row in self.df.iterrows():
            image_file = f'{self.data_dir}/{row.stage}_{self.resolution}/{row.StudyInstanceUID}.jpg'
            try:
                image = Image.open(image_file)
                yield np.array(image)
            except:
                if verbose:
                    print(f'Failed to load {image_file}')
                yield np.zeros((self.resolution, self.resolution), dtype=np.uint8)

    def fit_transform(self):
        pca_features = []
        for step in ['fit', 'transform']:
            if step == 'fit':
                print('Fitting PCA for feature extraction...')
            elif step == 'transform':
                print('Applying PCA for feature extraction...')
            batches = batch_generator(self.image_generator(), self.batch_size)
            num_batches = (self.df.shape[0] - 1) // self.batch_size + 1
            for batch in tqdm(batches, total=num_batches):
                cnn_in = self.cnn_preprocess_images(batch)
                cnn_out = self.cnn_extract_features(cnn_in)
                if step == 'fit':
                    self.pca.partial_fit(cnn_out)
                elif step == 'transform':
                    pca_features.append(self.pca.transform(cnn_out))
        # Add the transformed features to the main dataframe.
        pca_features = np.concatenate(pca_features, axis=0)
        pca_col_names = [f'pca_{i}' for i in range(pca_features.shape[1])]
        pca_features = pd.DataFrame(pca_features, columns=pca_col_names, index=self.df.index)
        self.df = pd.concat([self.df, pca_features], axis=1)
        return self.df.sort_index()

    def cnn_preprocess_images(self, image_data):
        """Prepare raw grayscale images for CNN."""
        image_data = torch.from_numpy(image_data[:, None, :, :])
        image_data = image_data.repeat_interleave(3, 1)
        image_data = image_data.float()
        image_data = self.normalizer(image_data)
        image_data = image_data.cuda()
        return image_data

    def cnn_extract_features(self, cnn_in):
        with torch.no_grad():
            output = self.cnn(cnn_in)
            return output.detach().cpu().numpy()
