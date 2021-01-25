from PIL import Image
from os import path

LOCATIONS_DICT = {
    'train images': 'train',
    'test images': 'test',
    'train csv': 'train.csv',
    'test csv': 'test.csv'
}

STAGES = ['train', 'test']


class DataIngestionManager:
    """Manages the process of downloading the dataset and resizing images."""

    def __init__(self, data_dir='data', resolution=224):
        self.data_dir = data_dir
        self.resolution = resolution
        self.data_verifier = DataVerifier(data_dir)

    def ingest(self):
        """
        Begin the process of data ingestion.

        Returns True if the process is sucessful, otherwise returns False.
        """
        # First, verify that the base dataset is present.
        if self.data_verifier.is_data_missing():
            self.data_verifier.download_missing_data()
            return False
        # Next,


class DataVerifier:
    """Object to verify that the data exists and download missing data."""

    def __init__(self, data_dir):
        """
        Verifies that the dataset exists in the filesystem.

        Keyword arguments:
            data_dir -- root directory where the dataset is stored
        """
        self.data_dir = data_dir
        self.verification_dict = self.verify_dataset()

    def is_data_missing(self):
        """Check whether any data is missing, and return a bool."""
        return any(self.verification_dict.values())

    def download_missing_data(self):
        """Print a message instructing you to download the missing data."""
        for key, exists in verification_dict.iteritems():
            if not exists:
                print(f'Could not find {self.get_location(key)}! Please download {key} from kaggle.com')

    def get_location(self, key):
        """Converts a location key into the full location."""
        return self.data_dir + '/' + LOCATIONS_DICT[key]

    def verify_dataset(self):
        """
        Verify that all parts of the base dataset exist in the filesystem.

        Return true if they all exist, and return false if they don't. If parts of the dataset are missing, print messages
        about which parts.
        """
        verification_dict = {}
        for key in LOCATIONS_DICT.keys():
            full_location = self.get_location(key)
            verification_dict[key] = path.exists(full_location)
        return verification_dict


class ResolutionManager:
    """Manages the images with different resolutions."""
    def __init__(self, resolution):
        self.resolution = resolution

    def get_resolution_locations(self, resolution, stage):
        """Return the location of the dataset with specified resolution."""
        return f'{self.data_dir}/{stage}_{resolution}'

    def verify_resolution_dataset(self):
        for stage in STAGES:
            location = self.get_resolution_locations((stage, self.resolution))
            if not path.exists(location):
                self.convert_resolution()

    def convert_resolution(self, stage):
        source =
        target = self.get_resolution_locations()



class ImageReader:
    """Provides access to the raw images"""

    def __init__(self, resolution):
        self.resolution = resolution

    def