import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    #This defines the structure of the NN.

    """
    WRITE IT AGAIN SUCH THAT IT ACHIEVES
    99.4% validation accuracy against MNIST
    Less than 20k Parameters
    You can use anything from above you want. 
    Less than 20 Epochs
    Have used BN, Dropout,
    (Optional): a Fully connected layer, have used GAP
    """

    def __init__(self):
        super(Net, self).__init__()

        drop_out_value = 0.05

        # Input Block

        self.conv1 = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Input = 28x28x1 | Output = 26x26x16 | RF = 3x3

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Input = 26x26x16 | Output = 24x24x16 | RF = 5x5

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Input = 24x24x16 | Output = 22x22x16 | RF = 7x7

            nn.MaxPool2d(2, 2),

            # Input = 22x22x16 | Output = 11x11x16 | RF = 14x14
        )

        # CONVOLUTION BLOCK 1

        self.conv2 = nn.Sequential(

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1,1), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Dropout(drop_out_value),

            # Input = 11x11x16 | Output = 11x11x8 | RF = 14x14

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Dropout(drop_out_value),

            # Input = 11x11x16 | Output = 9x9x16 | RF = 16x16

        )

        self.conv3 = nn.Sequential(

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Dropout(drop_out_value),

            # Input = 9x9x16 | Output = 7x7x16 | RF = 18x18

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Dropout(drop_out_value),

        )

        self.conv4 = nn.Sequential(

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Dropout(drop_out_value),

            # Input = 7x7x16 | Output = 7x7x16 | RF = 20x20

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Dropout(drop_out_value),

            # Input = 7x7x16 | Output = 5x5x16 | RF = 22x22

        )

        self.gap = nn.AvgPool2d(kernel_size=5)

        self.fc = nn.Linear(in_features=16, out_features=10, bias=False)

    def forward(self, x):
            
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
    
            x = self.gap(x)

            x = x.view(-1, 16)

            x = self.fc(x)

            return F.log_softmax(x, dim=-1)
