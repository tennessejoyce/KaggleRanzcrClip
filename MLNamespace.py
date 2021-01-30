import numpy as np
import pandas as pd
import torch
import torchvision.models
from torchvision.transforms import Normalize
from sklearn.decomposition import IncrementalPCA
from copy import copy
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def read_image(image_file, resolution):
    """Reads in the xray image from a file and returns an nparray."""
    try:
        image = Image.open(image_file)
        return np.array(image)
    except:
        print(f'Failed to load {image_file}')
        return np.zeros((resolution, resolution), dtype=np.uint8)


def assign_folds(counts_df, k):
    """Assign patients to folds while balancing the number of samples (not patients) per fold."""
    assignments = {}
    totals = [0] * k
    for key, value in counts_df.sort_values().iteritems():
        i_min = np.argmin(totals)
        assignments[key] = i_min
        totals[i_min] += value
    return pd.Series(assignments)


class MLNamespace:
    def __init__(self):
        pass

    def batch_apply(self, batch_size, source_cols, target_cols, shuffle=True):
        """
        Splits a namespace into batches so that an operation can be applied, then recombines the results.

        Keyword arguments:
            batch_size -- The maximum batch_size. If the data cannot be divided evenly, the last batch will be smaller.
            source_cols -- One or more pandas dataframes in the self to be split. They must all share the same index.
            target_cols -- Columns in the target self containing the result of the operation once it is performed.
            shuffle -- Whether to shuffle the index before splitting into batches.

        """
        index = getattr(self, source_cols[0]).index
        if shuffle:
            index = np.random.permutation(list(index))
        num_batches = (len(index) - 1) // batch_size + 1
        # Dictionary to store the results, as they are computed.
        results = {col: [] for col in target_cols}
        for index_batch in np.array_split(index, num_batches):
            batch = copy(self)
            # Split the pandas dataframes in source_cols into batches. Pass everything else unchanged.
            for col in source_cols:
                setattr(batch, col, getattr(batch, col).loc[index_batch])
            # Yield the batched self so that an operation can be performed on it.
            yield batch
            # Save the results of the operation.
            for col in target_cols:
                results[col].append(getattr(batch, col))
        # Combine the results, and add them to the original self.
        for col in target_cols:
            setattr(self, col, pd.concat(results[col]))


    # Methods for feature extraction


    def load_cnn_headless(self, architecture):
        """Load a convolutional neural network of specified architecture into self['cnn']."""
        assert hasattr(torchvision.models, architecture), f'CNN architecture not found {architecture}'
        # Load the CNN with pretrained weights.
        self.cnn = getattr(torchvision.models, architecture)(pretrained=True)
        # Replace the last fully connected layer with the indentity operation.
        self.cnn.fc = torch.nn.Identity()
        self.cnn.float().cuda()

    def cnn_preprocess_images(self):
        """Prepare raw grayscale images for CNN."""
        image_data = self.images
        image_data = torch.from_numpy(image_data[:, None, :, :])
        image_data = image_data.repeat_interleave(3, 1)
        image_data = image_data.float()
        if not hasattr(self, 'normalizer'):
            self.normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_data = self.normalizer(image_data)
        image_data = image_data.cuda()
        self.cnn_in = image_data

    def cnn_extract_features(self):
        self.cnn.eval()
        with torch.no_grad():
            output = self.cnn(self.cnn_in)
            self.cnn_out = output.detach().cpu().numpy()

    def instantiate_incremental_pca(self, whiten=False):
        self.pca = IncrementalPCA(whiten=whiten)

    def fit_incremental_pca(self,):
        self.pca.partial_fit(self.pca_in)

    def transform_incremental_pca(self):
        pca_out = self.pca.transform(self.pca_in)
        self.pca_features = pd.DataFrame(pca_out, index=self.df.index)

    # Methods for loading data

    def read_csv(self, name, filename, **kwargs):
        """Read tabular training self into self['train_df']."""
        setattr(self, name, pd.read_csv(filename, **kwargs))

    def load_train_csv(self):
        self.read_csv('df', f'{self.data_dir}/train.csv', index_col=0)

    def load_images(self, stage, resolution):
        directory = f'{self.data_dir}/{stage}_{resolution}'
        images = []
        for instance_id in self.df.index:
            image_file = f'{directory}/{instance_id}.jpg'
            images.append(read_image(image_file, resolution))
        self.images = np.stack(images)

    # Methods for machine learning

    def postprocess_features(self):
        self.pca_features = self.pca_features.sort_index()
        self.df = self.train_df.sort_index()

    def split_features(self):
        self.X = self.pca_features.values[:, :self.max_pca_features]
        self.patient_id = self.df.pop('PatientID')
        self.y = self.df.values

    def load_model(self, name, **kwargs):
        if name == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(**kwargs)
        elif name == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**kwargs)
        else:
            raise Exception(f'Unrecognized model: {name}.')

    def fit_model(self):
        self.model.fit(self.X, self.y)

    def evaluate_model(self):
        y_prob = self.model.predict_proba(self.X)[:, 1]
        self.evaluation_score = roc_auc_score(self.y, y_prob)
        return self.evaluation_score

    def predict_proba(self):
        y_prob = self.model.predict_proba(self.X)[:, 1]
        return pd.Series(y_prob, index=self.patient_id.index)

    def set_train_and_val(self, train_mask, val_mask):
        train, test = copy(self), copy(self)
        train.df = train.df[train_mask]
        test.df = test.df[val_mask]
        return train, test

    def train_val_split(self):
        """Splits the namespace into two, for training and validation."""
        # First, decide the indices for the split.
        unique_ids = self.patient_id.unique()
        val_size = int(self.val_fraction * len(unique_ids))
        rng = np.random.default_rng(self.random_state)
        val_patients = rng.choice(unique_ids, val_size)
        val_mask = self.patient_id.isin(val_patients)
        train_mask = ~val_mask
        return self.set_train_and_val(train_mask, val_mask)


    def kfold_cross_validation(self):
        """Implements k-fold cross validation."""
        # Series that assigns each patient id an integer from 0 to k-1.
        assignments = assign_folds(self.patient_id.value_counts(), self.cross_val_k)
        for i in range(self.cross_val_k):
            # Find patient ids in the ith fold.
            val_patients = assignments.index.where(assignments == i)
            # Use that to create a boolean mask over the samples.
            val_mask = self.patient_id.isin(val_patients)
            train_mask = ~val_mask
            # Use the mask to split into train and validation.
            yield self.set_train_and_val(train_mask, val_mask)


    def multitarget_split(self):
        """Splits a multi-target binary classification problem into many separate binary problems."""
        for i in range(self.df.shape[1]):
            binary_problem = copy(self)
            binary_problem.y = binary_problem.y[:, i]
            yield  binary_problem



    def feature_extraction_cnn_pca(self):
        self.load_train_csv()
        self.load_cnn_headless( architecture='resnet18')
        self.instantiate_incremental_pca()
        for stage in ['fit', 'transform']:
            target_cols = [] if stage == 'fit' else ['pca_features']
            num_batches = (self.df.shape[0] - 1) // self.batch_size + 1
            batches = self.batch_apply(batch_size=self.batch_size, source_cols=['df'], target_cols=target_cols)
            for batch in tqdm(batches, total=num_batches):
                batch.load_images('train', 224)
                batch.cnn_preprocess_images()
                batch.cnn_extract_features()
                batch.pca_in = batch.cnn_out
                if stage == 'fit':
                    batch.fit_incremental_pca()
                else:
                    batch.transform_incremental_pca()

    def model_training_split(self):
        """Train a model and evaluate on a validation set."""
        self.postprocess_features()
        train_scores = []
        test_scores = []
        for i, target in enumerate(tqdm(self.multitarget_split(), total=self.y.shape[1])):
            train, test = target.train_test_split()
            train.split_features()
            test.split_feautures()
            train.fit_model()
            train_score = train.evaluate_model()
            test_score = test.evaluate_model()
            train_scores.append(train_score)
            test_scores.append(test_score)
        for i, (train_score, test_score) in enumerate(zip(train_scores,test_scores)):
            print(f'Target {i}:  {train_score:.4f}  {test_score:.4f}')
        print(f'Average:  {np.mean(train_score):.4f}  {np.mean(test_score):.4f}')

    def model_training_kfold(self):
        """Train a model and evaluate with k-fold cross validation."""
        self.postprocess_features()
        self.scores = []
        for i, target in enumerate(tqdm(self.multitarget_split(), total=self.df.shape[1])):
            target.out_of_fold_probabilities = []
            for train, test in target.kfold_cross_validation():
                train.split_features()
                test.split_feautures()
                train.fit_model()
                test.out_of_fold_probabilities.append(test.predict_proba())
            target.out_of_fold_probabilities = pd.concat(target.out_of_fold_probabilities, axis=0)
            self.scores.append(roc_auc_score(target.y, target.out_of_fold_probabilities))
        for i, score in enumerate(self.scores):
            print(f'Target {i}:  {score:.4f}')
        print(f'Average:  {np.mean(self.scores):.4f}')

