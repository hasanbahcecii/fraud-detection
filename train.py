import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from preprocessing import train_loader # dataset
from transformer_model import TransformerClassifier # model

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create the model
model = TransformerClassifier()
model = model.to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

num_epochs = 2
# train loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc= f"Epoch {epoch + 1} / {num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs) # predictions
        loss = criterion(outputs, labels)
        loss.backward() # backpropagation (calculate grad)
        optimizer.step() # update weights

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1) # take the most probable class
        
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(train_loader)    # mean loss
    epoch_acc = correct / total
    print(f"Epoch {epoch + 1}/ {num_epochs} - Loss: {epoch_loss} - Accuracy: {epoch_acc}")

# save model 
torch.save(model.state_dict(), "transformer_fraud_model.pth")
print("Model saved succesfully.")