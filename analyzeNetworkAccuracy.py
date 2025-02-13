import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#2 hidden layers
#2 inputs, 2 hidden with 5 neurons each
class SimpleNet(nn.Module):
    # 2 because a, b
    input_size = 2
    # num of hidden neurons
    hidden_size = 5
    output_size = 1
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        # Right now RELU but we can change later
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out



data_path = './data/simpleReg.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at {data_path}")

data = pd.read_csv(data_path)
X = data[['a', 'b']].values.astype(np.float32)
y_true = data['y'].values.astype(np.float32)

# Convert to torch tensors (y_true reshaped to [N,1])
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y_true).unsqueeze(1)

networks_file = "WorkingNetworks/working_networks.pt"
if not os.path.exists(networks_file):
    raise FileNotFoundError(f"Networks file not found at {networks_file}")

networks_state_dicts = torch.load(networks_file, map_location=torch.device('cpu'))

mse_list = []
mae_list = []
r2_list = []

print("Evaluating networks on the test data:")
for net_name, state_dict in networks_state_dicts.items():
    model = SimpleNet()
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).squeeze().numpy()  # predictions shape: (N,)
    mse = mean_squared_error(y_true, predictions)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    mse_list.append(mse)
    mae_list.append(mae)
    r2_list.append(r2)
    print(f"{net_name}: MSE = {mse:.4f}, MAE = {mae:.4f}, R2 = {r2:.4f}")

print("\nSummary Metrics Across Networks:")
print(f"Average MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
print(f"Average MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
print(f"Average R2:  {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")

# Create a grid over the input space using the min/max of a and b from the data
a_min, a_max = data['a'].min(), data['a'].max()
b_min, b_max = data['b'].min(), data['b'].max()
a_range = np.linspace(a_min, a_max, 50)
b_range = np.linspace(b_min, b_max, 50)
A, B = np.meshgrid(a_range, b_range)
grid_inputs = np.column_stack([A.ravel(), B.ravel()]).astype(np.float32)
grid_tensor = torch.tensor(grid_inputs)

first_key = list(networks_state_dicts.keys())[0]
model = SimpleNet()
model.load_state_dict(networks_state_dicts[first_key])
model.eval()
with torch.no_grad():
    grid_predictions = model(grid_tensor).squeeze().numpy()
grid_predictions = grid_predictions.reshape(A.shape)

def target_function(a, b):
    return (1/5) * a**2 - (1/10) * b**3

Z_true = target_function(A, B)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create proxy artists for the legend with specific colors instead of colormaps
fake2Dline1 = plt.Rectangle((0,0), 1, 1, fc='blue', alpha=0.5)
fake2Dline2 = plt.Rectangle((0,0), 1, 1, fc='red', alpha=0.5)

# Update the surface plots to use the same colors as the legend
surf1 = ax.plot_surface(A, B, grid_predictions, color='blue', 
                       edgecolor='none', alpha=0.5, label='Network Prediction')
surf2 = ax.plot_surface(A, B, Z_true, color='red', 
                       edgecolor='none', alpha=0.5, label='True Function')

# Add title and labels
ax.set_title('Network Prediction vs True Function')
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('z')

# Add legend
ax.legend([fake2Dline1, fake2Dline2], ['Network Prediction', 'True Function'], 
         loc='upper left')

plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/network_regression_fit.png')