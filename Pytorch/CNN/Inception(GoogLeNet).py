import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels,out_1x1,red_3x3,out_3x3,red_5x5,out_5x5,out_pool):
        super().__init__()

        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size = 1),
            nn.ReLU(inplace = True)
        )

        # 1x1 to 3x3 branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels,red_3x3, kernel_size = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size =3,padding = 1),
            nn.ReLU(inplace=True)
        )

        # 1x1 to 5x5 branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels,red_5x5, kernel_size = 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size =5,padding = 2),
            nn.ReLU(inplace=True)
        )

        # pooling  to 1x1 branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size =3, stride = 1, padding = 1),
            nn.Conv2d(in_channels, out_pool, kernel_size = 1),
            nn.ReLU(inplace = True)
        )

    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # concatenate  along channel dimension
        return torch.concat([b1,b2,b3,b4], dim = 1)
    

# Example For a CIFAR-10 
x = torch.randn(1, 192, 32, 32)

inception = InceptionBlock(
    in_channels=192, # Input  → (N, 192, 32, 32) # Input channels
    out_1x1=64,      # Output → (N, 64,  32, 32) # for branch1 1x1 block
    red_3x3=96,      # (N, 192, 32, 32) → (N, 96, 32, 32) for branch2 1x1 block
    out_3x3=128,     # (N, 96, 32, 32) → (N, 128, 32, 32) for branch2 3x3 block
    red_5x5=16,      # (N, 192, 32, 32) → (N, 16, 32, 32) for branch3 1x1 block
    out_5x5=32,      # (N, 16, 32, 32) → (N, 32, 32, 32) for branch3 5x5 block
    out_pool=32      # (N, 192, 32, 32) → (N, 32, 32, 32) for branch4 maxpool 3x3 block with 32 output channels 
)

y = inception(x)
print(y.shape)
