import torch
import torch.nn as nn

# build a simple convolutional neural network for character recognition
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.build_model()

    def build_model(self):
        self.model = nn.Sequential()

        self.model.add_module('conv1', nn.Conv2d(1, 32, kernel_size=3, padding=1))
        self.model.add_module('relu1', nn.ReLU())
        self.model.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))

        self.model.add_module('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1))
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('pool2', nn.MaxPool2d(kernel_size=2, stride=2))

        self.model.add_module('flatten', nn.Flatten())
        self.model.add_module('fc1', nn.Linear(64 * 7 * 7, 128))
        self.model.add_module('relu3', nn.ReLU())
        self.model.add_module('dropout', nn.Dropout(0.25))
        self.model.add_module('fc2', nn.Linear(128, 36))  # 36 classes: 0-9 and A-Z
        

    def forward(self, x):
        return self.model(x)