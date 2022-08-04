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
from torchvision.models import resnet18
from torchvision import datasets, models, transforms
import os
import time

def read_args():
    parser = argparse.ArgumentParser(description="command line args for testing\
        the classifier")
    parser.add_argument("--weights", type = str, help= "path to weight file")
    parser.add_argument("--speed", action = "store_true", help="calcuate speed of the classifier")
    parser.add_argument("--test_data", type = str, help= "path to test data dir")
    parser.add_argument("--all_classes_accuracy", action= "store_true", help="check all classes accuracy")
    parser.add_argument("--img", type = int, help= "image resolution")
    parser.add_argument("--batch", type=int, help = 'batch size')
    parser.add_argument("--workers", type = int, help= "number of workers")
    parser.add_argument("--display_samples", action = "store_true", help= "display some images for visualization from test set..")
    parser.add_argument("--pretrained", action= "store_true", help="pretrained model")
    return parser.parse_args()


def load_data(test_data, img, batch=32, workers=8):
    """load test data for evaluation purposes
    args: test_data (path)
          Load the test data to evaluate the classifier performance on
          img (int)
          height and width of an image
    return: test_loader (batched data)
            return test loader with dataset in batches"""
    # transform test data
    transforms_test = transforms.Compose([
        transforms.Resize((img, img)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    global test_datasets
    test_datasets = datasets.ImageFolder(test_data, transforms_test)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch, shuffle = False, num_workers = workers)
    return test_loader

def imshow(images, predictions):
    """subplot each prediction with label
    args: images (batched data)
          predictions (batch predictions)"""
    
    classes = ("female", "male")
    plt.figure(figsize=(10, 10))
    for i,img in enumerate(images):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.cpu().numpy()
        plt.subplot(2, 8, i+1)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.xticks([])
        plt.yticks([])
        plt.title(f"{classes[predictions[i]]:5s}")
    plt.show()

def visualize_samples(images, predictions):
    '''visulize some samples from the test set
    args: test_loader (batched data)
          test set data in batches
        '''
    # dataiter = iter(test_loader)
    # images, labels = dataiter.next()

    # print images
    imshow(images, predictions)
    # print('GroundTruth: ', ' '.join(f'{classes[predictions[j]]:5s}' for j in range(16)))

def test_all_data(testloader, model, device, speed = False, all_classes=True):
    '''test all data set accuracy
    args: test_loader (batched data)
    return accuracy on all data
    '''
    classes = ("female", "male")
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    correct = 0
    total = 0
    count = 0
    time_accu = 0
# since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            count+=1
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            tic = time.time()
            outputs = model(images)
            toc = time.time()
            duration = toc - tic
            time_accu += duration
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if all_classes:
                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1
    # print accuracy for each class
    if all_classes:
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    avg_speed = time_accu/count
    accuracy = 100 * correct / total
    if speed == True:
        print(f'Average speed of the model: {avg_speed}')
    print(f'Accuracy of the network on the test images: {accuracy:.3f} %')

#merged with all_data accuracy calcuation function
#feel free to comment
def individual_class_acc(testloader, model):
    '''check accuracy for invidual classes
    args: dataloader (batched test data)
          model (CNN model)
        '''
    classes = ("female", "male")
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

def main():
    '''all managed code goes here'''
    args = read_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device:  {device}')
    # our custom model
 
    print("importing model!!\n")
    if args.pretrained:
        net = resnet18(pretrained=True)
        in_ftrs = net.fc.in_features
        net.fc = nn.Linear(in_ftrs, 2)
        net = net.to(device)
    else:

        net = custom_model(num_classes=2, img_shape= args.img).to(device)
        # load the trained weights 
        net.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')))
        print("weights loaded into the model!!\n")

    print('Now laoding test data!!\n')
    test_loader = load_data(args.test_data, args.img, args.batch)
    print('done!!\n')
    
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)
    outputs = net(images)
    skip, predicted = torch.max(outputs, 1)
    classes = ("female", "male")
    # print(type(images), type(predicted))
    # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
    #                           for j in range(16)))
    visualize_samples(images, predicted)
    time.sleep(5)
    print('checking accuracy on all data!!\n')
    if not args.speed and not args.all_classes_accuracy:
        acc = test_all_data(test_loader, net, device, False, False)
    elif not args.speed and args.all_classes_accuracy:
        acc = test_all_data(test_loader, net, device, False, True)
    elif args.speed and not args.all_classes_accuracy:
        acc = test_all_data(test_loader, net, device, True, False)
    else:
        acc = test_all_data(test_loader, net, device, True, True)
    print('done!!!\n')

if __name__ == "__main__":
    main()
