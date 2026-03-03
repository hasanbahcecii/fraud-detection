import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# load data
X = np.load("X_fraud.npy")
y = np.load("y_fraud.npy")

# normalization (z score)
scaler = StandardScaler()

# (1500, 20 , 5) -> (1500x20, 5)
X_reshaped = X.reshape(-1, X.shape[2]) # (30000, 5) 

# fit the scaler for all the data
X_scaled = scaler.fit_transform(X_reshaped)

# to original dimension
X = X_scaled.reshape(X.shape)

# convert to the original dimension
# split the data to %80 train, %20 tesst 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42, stratify= y)

# to tensor
X_train_tensor = torch.tensor(X_train, dtype= torch.float32)
X_test_tensor = torch.tensor(X_test, dtype= torch.float32)
y_train_tensor = torch.tensor(y_train, dtype= torch.long)
y_test_tensor = torch.tensor(y_test, dtype= torch.long)

# create dataloader
# create dataset using tensor
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size= batch_size)

# control: print shape of the data 
print(f"Train shape: {X_train_tensor.shape}")
print(f"Test shape: {X_test_tensor.shape}")