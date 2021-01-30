from glob import glob
from PIL import Image
import os
from tqdm import tqdm


def resize_images(source_dir, target_dir, target_resolution):
    """Extract images from one directory, resize, and write to another directory."""
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    for in_file in tqdm(glob(f'{source_dir}/*.jpg')):
        try:
            image = Image.open(in_file)
            image = image.resize((target_resolution, target_resolution), Image.BILINEAR)
            image_name = in_file.split('\\')[-1]
            out_file = f'{target_dir}/{image_name}'
            image.save(out_file)
        except Exception as e:
            print(f'Failed to open {in_file}')
            print(e)

def resize_all(data_dir, target_resolution):
    """Resizes both train and test images to a target resolution."""
    for stage in ['train', 'test']:
        print(f'Resizing {stage}...')
        resize_images(source_dir=f'{data_dir}/{stage}',
                      target_dir=f'{data_dir}/{stage}_{target_resolution}',
                      target_resolution=target_resolution)