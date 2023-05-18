import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper parameters

num_epochs = 25
batch_size = 64
learning_rate = 0.001


# MNIST
train_dataset = torchvision.datasets.CIFAR10(root="./data",train=True,transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.CIFAR10(root="./data",train=False,transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)


examples = iter(train_loader)
samples,labels = next(examples)
print(samples.shape,labels.shape)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0],cmap="gray")

#plt.show()

class ConvNet(nn.Module):

    def __init__(self,):
        super(ConvNet,self).__init__()

        self.conv1 = nn.Conv2d(3,9,5)
        self.conv2 = nn.Conv2d(9,12,5)
        
        self.pool1 = nn.MaxPool2d(2,2)
        self.pool2 = nn.MaxPool2d(2,2)
   
        self.l1 = nn.Linear(12*5*5,120)
        self.l2 = nn.Linear(120,60)

        self.l3 = nn.Linear(60,10)

    def forward(self,x):
        out = self.pool1(F.relu(self.conv1(x)))
        out = self.pool2(F.relu(self.conv2(out)))
    
  
        out = out.view(-1,12*5*5)
        out = F.relu(self.l1(out))
        out = F.relu(self.l2(out))
   
        out = self.l3(out)
        return out

model = ConvNet().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

# training loop

n_total_steps = len(train_loader)
for epoch in range(num_epochs):

    for i,(images,labels) in enumerate(train_loader):
       

        images = images.to(device)
        labels = labels.to(device)

        # forward

        outputs = model(images)

        loss = criterion(outputs,labels)

        # backward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}")


# test
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    for images,labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # value, index
        _, predictions = torch.max(outputs,1)
        n_samples += labels.shape[0]

        n_correct += (predictions==labels).sum().item()

    acc = 100.0 * n_correct / n_samples

    print(f"acc = {acc}")
