from models import VGG16, ResNet, Linear
import torch
from data import DBSCANBatchDataset, CSVDataset, FolderDataset
from train import BasicTraining, DistillationTraining
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torchvision
# from torchvision.models import vgg16, VGG16_Weights
from loss import Distillation_Loss
from torch.nn import CrossEntropyLoss
from infer import test

# model = VGG16(3, 10).to('cuda:0')
# sample = torch.randn(1, 3, 224, 224).to('cuda:0')
# print(model(sample).shape)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.Resize((224, 224))])

model = VGG16(3, 10)
model.load_state_dict(torch.load('pretrained/distillation_4.pth'))
test(model, 'test/', transform)

# batch_size = 4

# trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True,
#                                         download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=0)

# testset = torchvision.datasets.CIFAR10(root='./dataset', train=False,
#                                        download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=0)

# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# teacher_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True).to('cuda:0')
# model = VGG16(3, 10).to('cuda:0')

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# # criterion = Distillation_Loss(0.9, 0.5)
# criterion = CrossEntropyLoss()
# # training = DistillationTraining(model, teacher_model, trainloader, testloader, optimizer, criterion, 'cuda:0', pretrained_model_path='distillation_4.pth', num_epochs=5)
# training = BasicTraining(model, trainloader, testloader, optimizer, criterion, 'cuda:0', num_epochs=5)
# training.train()

# data = CSVDataset('iris.csv')
# train, val = train_test_split(data, test_size=0.2)
# model = Linear(4, 3, [10, 10])
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = torch.nn.CrossEntropyLoss()
# training = BasicTraining(model, train, val, optimizer, criterion, 'cpu')
# training.train()

# data = DBSCANBatchDataset('iris.csv', batch_size=8, attrs=["sepal_length", "sepal_width"])
# print(data[0])
# print(pd.unique(data.df['batch_label']))
# print(data.df[data.df.keys()])
# # visualize outputs
# colors = data.df['batch_label'].astype(int)
# plt.scatter(data.df["sepal_length"], data.df["sepal_width"], c = colors)
# plt.show()

# x = torch.randn(64, 3, 224, 224)
# print(model(x).shape)