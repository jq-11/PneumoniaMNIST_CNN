

import torch
import torch.nn as nn # tools to build and customize neural network
import torch.optim as optim # provides optimization algorithms to update parameters & minimize loss function
from torch.utils.data import DataLoader # batches/shuffles data
from torchvision import transforms # image preprocessing
import medmnist # provides small, medical image datasets
from medmnist import PneumoniaMNIST
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import numpy as np

# -------------------------------
# 1. Load Data
# -------------------------------

data_flag = 'pneumoniamnist'
download = True

# processing images before training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load labeled image samples
# label: 0 = healthy, 1 = has pneumonia
train_dataset = PneumoniaMNIST(split='train', transform=transform, download=download)
test_dataset  = PneumoniaMNIST(split='test', transform=transform, download=download)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# -------------------------------
# 2. Define CNN
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*7*7, 64)
        self.fc2 = nn.Linear(64, 1)  # binary classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32*7*7)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # outputs shape [batch,1]
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# -------------------------------
# 3. Train Model
# -------------------------------
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(device)
        labels = labels.view(-1, 1)  # match output shape

        optimizer.zero_grad()
        outputs = model(images)  # shape [batch,1]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# -------------------------------
# 4. Evaluate Model
# -------------------------------
model.eval()
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.float().to(device)
        labels = labels.view(-1, 1)  # shape [batch,1]

        outputs = model(images)  # shape [batch,1]
        preds = (outputs > 0.5).int()

        y_true.extend(labels.cpu().numpy().flatten())
        y_pred.extend(preds.cpu().numpy().flatten())
        y_prob.extend(outputs.cpu().numpy().flatten())

# -------------------------------
# 5. Metrics: Confusion Matrix + ROC
# -------------------------------
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Pneumonia"],
            yticklabels=["Normal", "Pneumonia"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

accuracy = np.mean(np.array(y_true) == np.array(y_pred))
print(f"Test Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
