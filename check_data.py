import argparse
import os
import sys
from random import shuffle
import shutil

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type = str, help= "dataset directory contating test, train, and validation directories")
    parser.add_argument('--unique', action="store_true", help="check for unique images in the dirs")
    parser.add_argument('--copy', action= "store_true", help="copy the data")
    return parser.parse_args()



def check_same_files(data, unique= True, copy=False):
    '''function that checks for same files in trian, val, and test dirs
    args: data (path) -> path to dataset
    return:'''
    # directories
    training = os.path.join(data, "Training")
    testing = os.path.join(data, "Testing")
    validation= os.path.join(data, "Validation")
    
    training_imgs = os.listdir(training)
    validation_imgs = os.listdir(validation)
    testing_imgs = os.listdir(testing)
    if unique == True:

        if len(set(training_imgs)) == len(training_imgs):
            print("unique content in training")
        if len(set(validation_imgs)) == len(validation_imgs):
            print("unique content in validation")
        if len(set(testing_imgs)) == len(testing_imgs):
            print("unique content in test set")
        print('unique test passed!!\n')
    if copy == True:

        ## copying data from evaluations dirs to training dir
        training_female = os.path.join(training, 'female')
        training_male = os.path.join(training, 'male')
        val_female = os.path.join(validation, 'female')
        val_male = os.path.join(validation, 'male')
        test_female = os.path.join(testing, 'female')
        test_male = os.path.join(testing, 'male')
        ##now copy data
        copy_data(training_female, val_female)
        copy_data(training_female, test_female)
        copy_data(training_male, val_male)
        copy_data(training_male, test_male)

def copy_data(destination, source):
    '''copy data from source directory to destination directory'''

    source_files = os.listdir(source)
    shuffle(source_files)
    src_size = len(source_files)
    img_to_move = src_size - 5000
    for img in source_files[:img_to_move]:
        src_file = os.path.join(source, img)
        des_file = os.path.join(destination, img)
        shutil.move(src_file, des_file)


def main():
    args = read_args()
    check_same_files(args.data, args.unique, args.copy)
    print("done!!\n")

if __name__ == "__main__":
    main()