import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import os


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(in_features=28 * 28, out_features=128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=128, out_features=64)
        self.layer3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='../src/data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='../src/data', train=False, download=True, transform=transform)

train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNISTClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        # Print every 100 batches
        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_loss = running_loss / 100
            accuracy = 100 * correct / total
            print(f'Epoch {epoch+1}, Batch {batch_idx * 64}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.1f}%')
            running_loss = 0.0

def evaluation(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

model_path = './src/models/mnist_classifier.pth'

if os.path.exists(model_path):
    print('Model found')
    print('Model will be load and evaluated')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    accuracy = evaluation(model, test_loader, device)
    print(f'Accuracy: {accuracy:.1f}%')
else:
    print('Model not found')
    print('Model will be trained')
    num_epochs = 10
    for epoch in range(num_epochs):
        train_epoch(model, train_loader, optimizer, criterion, device)
        accuracy = evaluation(model, test_loader, device)
        print(f'Epoch {epoch+1}, Accuracy: {accuracy:.1f}%')

    torch.save(model.state_dict(), model_path)



