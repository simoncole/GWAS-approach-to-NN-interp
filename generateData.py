import numpy as np
import pandas as pd

np.random.seed(12)

# Generate data for the fn
num_training_points = 300000
a = np.random.randn(num_training_points, 1)
b = np.random.randn(num_training_points, 1)

print(a.min(), a.max())

# Define the target function, in future could make dynamic
def target_function(a, b):
    return (1/5) * a**2 - (1/10) * b**3

# Get labels
y = target_function(a, b)

data = pd.DataFrame({
    'a': a.flatten(),
    'b': b.flatten(),
    'y': y.flatten()
})

#for now, ok to upload to git but if the data gets too large we'll want to use env and store it locally
# data.to_csv('./data/simpleReg.csv', index=False)