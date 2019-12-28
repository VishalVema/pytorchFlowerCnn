from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
# from torchvision.utils import make_grid

import torch.nn as nn

from torch.optim import Adam

import torch

flowers_path = './flowers'

transformations = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

total_dataset = datasets.ImageFolder(flowers_path, transform=transformations)
dataset_loader = DataLoader(dataset=total_dataset, batch_size=100)
items = iter(dataset_loader)
image, label = items.next()


def show_image(path):
    img = Image.open(path)
    img_arr = np.array(img)
    plt.figure(figsize=(5, 5))
    plt.imshow(np.transpose(img_arr, (0, 1, 2)))
    plt.show()


def show_transformed_image(image):
    np_image = image.numpy()
    plt.figure(figsize=(20, 20))
    plt.imshow(np.transpose(np_image, (1, 2, 0)))
    plt.show()


# show_transformed_image(make_grid(image))

# print(total_dataset.class_to_idx)

train_size = int(0.8 * len(total_dataset))
test_size = len(total_dataset) - train_size
train_dataset, test_dataset = random_split(
    total_dataset,
    [train_size, test_size]
)

train_dataset_loader = DataLoader(
    dataset=train_dataset,
    batch_size=100
)
test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=1000)


class FlowerClassifierCNNModel(nn.Module):

    def __init__(self, num_classes=5):
        super(FlowerClassifierCNNModel, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(
            in_channels=12,
            out_channels=24,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu2 = nn.ReLU()

        self.lf = nn.Linear(in_features=32 * 32 * 24, out_features=6)

    def forward(self, input):
        output = self.conv1(input)
        output = self.relu1(output)

        output = self.maxpool1(output)

        output = self.conv2(output)
        output = self.relu2(output)

        output = output.view(-1, 32 * 32 * 24)

        output = self.lf(output)

        return output


if torch.cuda.is_available():
    # you can continue going on here, like cuda:1 cuda:2....etc.
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

print(device)
cnn_model = FlowerClassifierCNNModel().to(device)
optimizer = Adam(cnn_model.parameters())
loss_fn = nn.CrossEntropyLoss()


def train_and_build(n_epoches):
    for epoch in range(n_epoches):
        cnn_model.train()
        print(str(epoch) + " Epoch")
        for i, (images, labels) in enumerate(train_dataset_loader):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            outputs = cnn_model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # print(str(i) + "train nth")


train_and_build(200)

cnn_model.eval()
test_acc_count = 0
for k, (test_images, test_labels) in enumerate(test_dataset_loader):
    test_images = test_images.to(device)
    test_labels = test_labels.to(device)
    test_outputs = cnn_model(test_images)
    _, prediction = torch.max(test_outputs.data, 1)
    test_acc_count += torch.sum(prediction == test_labels.data).item()

test_accuracy = test_acc_count / len(test_dataset)

print(test_accuracy)


# test_image = Image.open(flowers_path+"/dandelion/13920113_f03e867ea7_m.jpg")
# test_image_tensor = transformations(test_image).float()
# test_image_tensor = test_image_tensor.unsqueeze_(0)
# output = cnn_model(test_image_tensor)
# class_index = output.data.numpy().argmax()

# print(class_index)
# print(total_dataset.class_to_idx)
