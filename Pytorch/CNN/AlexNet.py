import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self,num_classes = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 11, stride = 4, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            nn.Conv2d(64, 192, kernel_size= 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            nn.Conv2d(192, 384, kernel_size = 3, padding = 1),
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size =3, stride = 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(),

            nn.Dropout(p = 0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),

            nn.Linear(4096,num_classes)
        )
    
    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x