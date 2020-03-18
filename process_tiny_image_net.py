'''
    This file is meant to process the TinyImageNet dataset into a format accepted by the ImageFolder
    dataset interface in torch vision. It will only need to be run once on any given machine this
    repo is being executed on.
'''

import shutil
import os

# Copy to directory:
#shutil.copy('sample1.txt', '/home/varung/test/sample2.txt')

# After first walk
# Iterate over all the files
# Make sure the file is not a .txt
# Find the class name of the image (to the right of the _ in the file name)
# Make the directory for this new class in the new dataset if it does not exist
# Copy the image to an appropriate folder in the new dataset

PROCESSED_DATASET_ROOT = './datasets/processed-tiny-imagenet'
RAW_DATA_ROOT = './datasets/tiny-imagenet-200/train'

if __name__ == '__main__':
    #if not os.path.exists('./datasets/processed-tiny-imagenet/n12267677/images/'):
        #os.makedirs('./datasets/processed-tiny-imagenet/n12267677/images/')

    #shutil.copy('./datasets/tiny-imagenet-200/train/n12267677/images/n12267677_0.JPEG', './datasets/processed-tiny-imagenet/n12267677/images/n12267677_0.JPEG')

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