import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm


# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 100
batch_size = 100
# learning_rate = 1e-3
learning_rate = 1e-2 # section 2 lr change


# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        # Section 1 + 2:
        # self.fc1 = nn.Linear(input_size, num_classes)
        # Section 3:
        self.fc0 = nn.Linear(input_size, 500)
        self.fc1 = nn.Linear(500, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Section 1 + 2:
        # out = self.fc1(x)
        # Section 3:
        out = self.fc0(x)
        out = self.relu(out)
        out = self.fc1(out)
        return out


net = Net(input_size, num_classes)


# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # section 2 optimizer change
losses = []
epoch_marks = []
# Train the Model
for epoch in tqdm(range(num_epochs)):
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        # Forward + Backward + Optimize
        predictions = net(images)
        loss = criterion(predictions, labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Mark the end of each epoch
    epoch_marks.append(len(losses))

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    outputs = net(images)
    _, predictions = torch.max(outputs, 1)
    correct += (predictions == labels).sum().item()
    total += labels.size(0)

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# Plot the training loss
plt.figure(figsize=(10,5))
plt.plot(losses)
# Add vertical lines for epoch boundaries
for mark in epoch_marks[:-1]:  # Skip the last mark as it's at the end
    plt.axvline(x=mark, color='r', linestyle='--', alpha=0.2)
plt.title(f'Section 3: {100 * correct / total:.2f} %')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss_section_3.png')
plt.close()

# Save the Model
torch.save(net.state_dict(), 'model.pkl')
