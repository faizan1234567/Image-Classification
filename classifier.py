## Import modules 
from PIL import Image
import time
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations
import albumentations.pytorch
from matplotlib import pyplot as plt
import cv2
import numpy as np
import argparse
from models import custom_model
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import datasets, models, transforms
import os

from torchvision.models import resnet18,  ResNet18_Weights
# check device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
print(f'accelartor is being used?: {device}')

#read commmand line arguments
def read_args():
    """read commnad line arguments"""
    parser = argparse.ArgumentParser(description= "adding commmand line args for second stage classifier")
    parser.add_argument("--epochs", type=int, help= "number of epochs to train the model for..")
    parser.add_argument("--weights", type = str, help= "weight file path, pth extension, or pretrained model weight name")
    parser.add_argument("--data", type = str, help = "dataset directory")
    parser.add_argument("--img", type = int, help= "image shpae to resize before training")
    parser.add_argument("--batch", type = int, help = "batch size")
    parser.add_argument("--workers", type = int, help= "number of workers")
    parser.add_argument("--classes", type = int, help= "number of classes")
    parser.add_argument("--pretrained", action='store_true', help="i.e. use resnet18 model or others in pytorch.org")
    parser.add_argument("--name",type = str, help= "pretrained model name")
    return parser.parse_args()

## load the dataset
def load_dataset(data_dir, img, batch_size, workers):
    """load the dataset for dataset directory and put this dataset into batches
    Arguments: data_dir (path)
               path to the dataset directory
               img (int)
               image shape to be resized by torch vision transforms before feeding it to the 
               classifier
               batch_size (int)
               batch size to be used to load data
               workers (int)
               number of workers ...
    Return:    train_dataset
               training dataset in batches
               val_dataset
               validation dataset in batches ..
                """
    transforms_train = transforms.Compose([
    transforms.Resize((img, img)),
    transforms.RandomHorizontalFlip(p=0.6), #add horizontal flip randonly
    transforms.RandomRotation(degrees = (0, 45)),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    #transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
    transforms.RandomResizedCrop(size=(img, img)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
    ])

    transforms_val = transforms.Compose([
        transforms.Resize((img, img)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # data_dir = './gender_classification_dataset'
    global train_datasets
    global val_datasets
    train_datasets = datasets.ImageFolder(os.path.join(data_dir, 'Training'), transforms_train)
    val_datasets = datasets.ImageFolder(os.path.join(data_dir, 'Validation'), transforms_val)

    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_dataloader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=True, num_workers=workers)

    print('Train dataset size:', len(train_datasets))
    print('Validation dataset size:', len(val_datasets))
    return train_dataloader, val_dataloader

def train(model, num_epochs, train_dataloader, optimizer,
          criterion, train_datasets, val_dataloader, val_datasets, schedular):
    """train classifier on dataset with num_epochs
    Argument: num_epochs
              train the classifier for num-epochs"""
    start_time = time.time()
    best_acc = 0
    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.
        running_corrects = 0
        
        # load a batch data of images
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward inputs and get output
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # get loss value and update the network weights
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            # print(running_loss)
        epoch_loss = running_loss / len(train_datasets)
        epoch_acc = running_corrects / len(train_datasets) * 100.
        schedular.step()
        print('Epoch: {} Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))

        model.eval()
        with torch.no_grad():
            running_loss = 0.
            running_corrects = 0

            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(val_datasets)
            epoch_acc = running_corrects / len(val_datasets) * 100.
            if best_acc < epoch_acc or epoch == num_epochs -1:
                if epoch == num_epochs -1:
                     save_path = 'last.pth'
                     torch.save(model.state_dict(), save_path)
                else:
                    best_acc = epoch_acc
                    contents = os.getcwd()
                    contents_list = os.listdir(contents)
                    for content in contents_list:
                        if content.endswith('.pth'):
                            os.remove(content)
                    save_path = f'{epoch}_epoch_face_gender_classification_transfer_learning_with_ResNet18.pth'
                    torch.save(model.state_dict(), save_path)
            print('Validation: {} Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))



def main():
    args = read_args()
    print("Loading the dataset!!\n")
    train_dataloader, val_dataloader = load_dataset(args.data, args.img, args.batch, args.workers)
    print("done!!!\n")

    print("loading the model and setting up model configuratin!!\n")
    

    # laod the model WITH THE LATEST Weights, please use model name with pytorch import convention
    # and model weight name should be like ResNet101 etc, for more see pytorch.org
    if args.pretrained:
        model = models.__dict__[args.name](weights = f"{args.weights}_Weights.DEFAULT")
        in_ftrs = model.fc.in_features
        model.fc = nn.Linear(in_ftrs, args.classes)
        model = model.to(device)
    else:
        model = custom_model(args.classes, args.img).to(device)
    #setting SGD optimizer, change to Adam if you want, you could also change learning rate by changing lr
    optimizer= optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    print("done!!!!\n")
    print("Now training!!")
    train(model, args.epochs, train_dataloader, optimizer,
          criterion, train_datasets, val_dataloader, val_datasets, exp_lr_scheduler)
    print("Training done!!")

if __name__ == "__main__":
    main()
