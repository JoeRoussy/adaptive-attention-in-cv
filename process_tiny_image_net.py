'''
    This file is meant to process the TinyImageNet dataset into a format accepted by the ImageFolder
    dataset interface in torch vision. It will only need to be run once on any given machine this
    repo is being executed on.
'''

import shutil
import os

PROCESSED_DATASET_ROOT = './datasets/processed-tiny-imagenet'
RAW_DATA_ROOT = './datasets/tiny-imagenet-200/train'

if __name__ == '__main__':
    for subdir, dirs, files in os.walk(RAW_DATA_ROOT):
        for file in files:
            if '.JPEG' in file:
                class_name = file.split('_')[0]
                processed_folder = PROCESSED_DATASET_ROOT + '/{}/'.format(class_name)

                if not os.path.exists(processed_folder):
                    os.makedirs(processed_folder)

                current_path = '{}/{}'.format(subdir, file)
                new_path = processed_folder + file
                
                print('Copying image at {} to {}'.format(current_path, new_path))

                shutil.copy(current_path, new_path)