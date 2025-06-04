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
    
class DropoutMLP(nn.Module):
    '''A simple MLP with 4 hidden layers, ReLU activation function, and dropout for uncertainty estimation'''
    def __init__(self, in_size, hidden_size, out_size, dropout_rate=0.5):
        super(DropoutMLP, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, input_size=10, latent_dim=(16, 16), output_dim=750, dropout_rate=0.3):
        super(CNN, self).__init__()
        # Fully connected layers to map the input parameters to a 2D latent space
        self.fc1 = nn.Linear(input_size, latent_dim[0] * latent_dim[1])  # Mapping to 2D latent space
        # Reshape the output to be 2D: (batch_size, 1, latent_dim[0], latent_dim[1])
        self.latent_dim = latent_dim
        # Convolutional layers with dilations and dropout
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=4, dilation=4)
        # Dropout layer after each convolutional layer
        self.dropout = nn.Dropout(dropout_rate)
        # Calculating the output size of the convolutional layers
        conv_output_dim = latent_dim[0]  # Keep output dimensions same due to padding
        self.fc2_input_size = 64 * conv_output_dim * conv_output_dim  # 64 channels from conv3
        self.fc2 = nn.Linear(self.fc2_input_size, output_dim)  # Output size is 750

    def forward(self, x):
        # Step 1: Map input parameters to 2D latent space
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, self.latent_dim[0], self.latent_dim[1])  # Reshape to 2D latent space
        # Step 2: Pass through convolutional layers with dropout
        x = F.relu(self.conv1(x))
        x = self.dropout(x)  # Apply dropout after conv1
        x = F.relu(self.conv2(x))
        x = self.dropout(x)  # Apply dropout after conv2
        x = F.relu(self.conv3(x))
        x = self.dropout(x)  # Apply dropout after conv3
        # Step 3: Flatten and fully connected layer
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_features)
        # Step 4: Fully connected output layer
        x = self.fc2(x)
        return x
    
class FastWeightCNN(nn.Module):
    def __init__(self, input_size=10, latent_dim=(16, 16), output_size=750, dropout_rate=0.3):
        super(FastWeightCNN, self).__init__()
        self.input_size = input_size
        # Fully connected layers to map the input parameters to a 2D latent space
        self.fc1 = nn.Linear(input_size, latent_dim[0] * latent_dim[1])
        self.latent_dim = latent_dim
        # Convolutional layers with dilations and dropout
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=4, dilation=4)
        # Dropout layer after each convolutional layer
        self.dropout = nn.Dropout(dropout_rate)
        conv_output_dim = latent_dim[0]  # Keep output dimensions same due to padding
        self.fc2_input_size = 64 * conv_output_dim * conv_output_dim  # 64 channels from conv3
        self.fc2 = nn.Linear(self.fc2_input_size, output_size)

    def forward(self, x, params=None, activations=None):
        if params is None:
            params = {name: param for name, param in self.named_parameters()}

        # Capture activation for fc1
        x = F.relu(F.linear(x, params['fc1.weight'], params['fc1.bias']))
        if activations is not None:
            activations['fc1'] = x
        x = x.view(-1, 1, self.latent_dim[0], self.latent_dim[1])

        # Capture activation for conv1
        x = F.relu(F.conv2d(x, params['conv1.weight'], params['conv1.bias'], padding=1, dilation=1))
        if activations is not None:
            activations['conv1'] = x
        x = self.dropout(x)

        # Capture activation for conv2
        x = F.relu(F.conv2d(x, params['conv2.weight'], params['conv2.bias'], padding=2, dilation=2))
        if activations is not None:
            activations['conv2'] = x
        x = self.dropout(x)

        # Capture activation for conv3
        x = F.relu(F.conv2d(x, params['conv3.weight'], params['conv3.bias'], padding=4, dilation=4))
        if activations is not None:
            activations['conv3'] = x
        x = self.dropout(x)

        # Flatten before the fully connected layer
        x = x.view(x.size(0), -1)

        # Capture activation for fc2
        x = F.linear(x, params['fc2.weight'], params['fc2.bias'])
        if activations is not None:
            activations['fc2'] = x

        return x

    def get_params(self):
        """
        Return a dictionary of all learnable parameters in the network.
        This will be used to pass fast weights in MetaLearner.
        """
        return {name: param for name, param in self.named_parameters()}

