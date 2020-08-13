# author lizhu
# date 2020-08-14
'''
    This is using pytorch'cnn to classify the MNIST data.
'''

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms


# Define Hyper parameters
NUM_CLASS = 16
BATCH_SIZE = 100
EPOCHS = 5
LEARNING_RATE = 0.01
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
train_dataset = dsets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())

test_dataset = dsets.MNIST(root='../data', train=False, download=True, transform=transforms.ToTensor())

# MNIST dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Convolutional neural network(two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_class=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(32*7*7, NUM_CLASS)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    
model = ConvNet(10).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Train the model
print("Start training...")
total_step = len(train_loader)
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        # To device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch+1, EPOCHS, i+1, total_step, loss.item()))
            
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print('The accuracy of model on 10000 test images: {} %'.format(100 * correct/total))
    
# Save the model
torch.save(model.state_dict(), 'cnn_2layer.ckpt')