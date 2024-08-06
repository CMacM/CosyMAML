import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA

class ResBlock(nn.Module):
    '''Residual block for the ResNet-like architecture. This block is used to
    provide a shortcut layer in the LINNA architechture'''
    def __init__(self, hidden_size):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out += residual  # Skip connection
        return F.relu(out)

class LINNA(nn.Module):
    '''LINNA architeture, see [arXiv:2203.05583v1]'''
    def __init__(self, in_size, hidden_size, out_size):
        super(LINNA, self).__init__()
        self.fc1 = nn.Linear(in_size, in_size * 2)
        self.fc2 = nn.Linear(in_size * 2, hidden_size)
        
        # Implementing the ResNet-like blocks
        self.res_blocks = nn.Sequential(
            ResBlock(hidden_size),
            ResBlock(hidden_size),
            ResBlock(hidden_size)
        )
        
        self.fc3 = nn.Linear(hidden_size, hidden_size // 8)
        self.fc4 = nn.Linear(hidden_size // 8, hidden_size // 2)
        self.fc5 = nn.Linear(hidden_size // 2, out_size)
        self.fc6 = nn.Linear(out_size, out_size)  # Final output layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Process through ResNet-like blocks
        x = self.res_blocks(x)
        
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        output = self.fc6(x)
        return output
    
class CONNECT(nn.Module):
    '''CONNECT architechture, see [arXiv:2205.15726v2]'''
    def __init__(self, in_size, hidden_size, out_size):
        super(CONNECT, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x
    
class SimpleMLP(nn.Module):
    '''A simple MLP with 4 hidden layers and ReLU activation function'''
    def __init__(self, in_size, hidden_size, out_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x
    
class PCADropoutMLP(nn.Module):
    '''A simple MLP with 3 hidden layers, ReLU activation function and dropout'''
    def __init__(self, in_size, hidden_size, out_size, dropout_rate):
        super(PCADropoutMLP, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc4 = nn.Linear(hidden_size//2, out_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x