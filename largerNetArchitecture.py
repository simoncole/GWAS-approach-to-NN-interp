import torch.nn as nn

# 2-32-64-64-32-1
# 2 inputs, 4 hidden layers with sizes [32, 64, 64, 32], output size 1
class LargerNet(nn.Module):
    input_size = 2  # Inputs: a, b
    hidden_sizes = [32, 64, 64, 32]  # Hidden layer sizes
    output_size = 1

    def __init__(self):
        super(LargerNet, self).__init__()
        # Fully connected layers
        self.fc1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.fc2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.fc3 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])
        self.fc4 = nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3])
        self.fc5 = nn.Linear(self.hidden_sizes[3], self.output_size)
        # Activation function
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = self.fc4(out)
        out = self.tanh(out)
        out = self.fc5(out)
        return out
