# ADDING THESE
import torch
import torch.nn as nn
from torchvision import models
from utils.torch_utils import select_device


# Please add mean and standard deviation of 2nd stage classifier's dataset
mean = [0.6260697307067842, 0.6012213494919662, 0.5740048955840032]
std = [0.2769153533724628, 0.2770797896447206, 0.28843602809799673]

class_names2 = None
super_class = None                  # The class whose output is not supposed to go to image classifier

def create_model(n_classes,device):
    # model = torch.hub.load('pytorch/vision:v0.6.0','resnext50_32x4d',pretrained=True)
    model = torch.hub.load('pytorch/vision:v0.6.0','resnet18',pretrained=True)
    # model = models.resnet18(pretrained=True)
    n_features = model.fc.in_features
    print(n_classes)
    model.fc = nn.Sequential(
        nn.Linear(n_features, n_classes),
        # nn.Softmax(dim=1)
    )
    print(model)

    return model#.to(device)

device='0'
device = select_device(device)
class_names2 = ["Male","Female","unknown"]
#
# modelc = create_model(len(class_names2),device)
# # TODO: Provide the path to load the pre-trained image classifier model .pt file below
# checkpoint = torch.load('/home/facit/Music/face_gender_classification_transfer_learning_with_ResNet18.pth')
# print(checkpoint)
# for key in checkpoint:
#     print(key)
#
# modelc.load_state_dict(checkpoint)
# # modelc.load_state_dict(checkpoint['model_state_dict'])
# modelc.to(device).eval()
#
# model.load_state_dict(torch.load(save_path))


save_path = r'set path here'

model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 3) # binary classification (num_of_class == 2)
model = torch.nn.DataParallel(model)

model.load_state_dict(torch.load(save_path))
model.to(device)