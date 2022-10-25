# import libraries
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionalNetwork(nn.Module):
    '''
        basic model for implementing a CNN with two conv layers
        parameters are hardcoded for now
    '''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3 ,1)      # in_channels, out_channels, kernel_size, stride=1, padding=0
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)   # (n - r) ^ 2 * channels
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, X):
        '''
            connects the layers in __init__ with given input X

            @param X, 24 x 24 image
        '''
        
        X = F.relu(self.conv1(X))   # can put in 1 line to improve readability
        X = F.max_pool2d(X, 2, 2)   # kernel, stride
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 5 * 5 * 16)  # transform the size before connecting to linear layer
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim = 1)
        