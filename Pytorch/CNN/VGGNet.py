import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels,out_channels, num_convs):
        super().__init__()

        layers = []
        for i in range(num_convs):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size = 3,
                    padding = 1
                )
            )
            
            layers.append(
                nn.ReLU(inplace = True)
            )
        layers.append(
            nn.MaxPool2d(
                kernel_size = 2, stride = 2
            )
        )

        self.block = nn.Sequential(*layers)

    def forward(self,x):
        return self.block(x)
    
class VGGNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            VGGBlock(3,   64, num_convs=2),   # Block 1
            VGGBlock(64, 128, num_convs=2),   # Block 2
            VGGBlock(128, 256, num_convs=3),  # Block 3
            VGGBlock(256, 512, num_convs=3),  # Block 4
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
