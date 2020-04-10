'''
    This file is meant to process the TinyImageNet dataset into a format accepted by the ImageFolder
    dataset interface in torch vision. It will only need to be run once on any given machine this
    repo is being executed on.
'''

import shutil
import os
import argparse


PROCESSED_DATASET_ROOT = './datasets/processed-tiny-imagenet'
RAW_DATA_ROOT = './datasets/tiny-imagenet-200/train'

parser = argparse.ArgumentParser('parameters')

# ATTENTION VARS
parser.add_argument('--num_classes', type=int, default=200,
                    help='number of classes to use')
parser.add_argument('--image_per_class', type=int, default=100,
                    help='number of images to use per class')

args = parser.parse_args()

if __name__ == '__main__':
    processed_subdirs = 0
    for subdir, dirs, files in os.walk(RAW_DATA_ROOT):
        processed_subdirs += 1
        img_count = 0
        for file in files:
            if '.JPEG' in file and img_count < args.image_per_class:
                img_count += 1
                class_name = file.split('_')[0]
                processed_folder = PROCESSED_DATASET_ROOT + '/{}/'.format(class_name)

                if not os.path.exists(processed_folder):
                    os.makedirs(processed_folder)

                current_path = '{}/{}'.format(subdir, file)
                new_path = processed_folder + file
                
                print('Copying image at {} to {}'.format(current_path, new_path))

                shutil.copy(current_path, new_path)

        if processed_subdirs == (args.num_classes * 2 + 1):
            break
