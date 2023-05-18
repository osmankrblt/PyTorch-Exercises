import torch
import torchvision


import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math


class WineDataset(Dataset):

    def __init__(self,transform=None):
        # data loading
        xy = np.loadtxt("data/wine.csv",delimiter=",",dtype=np.float32,skiprows=1)

        self.x = xy[:,1:]
        self.y = xy[:,[0]]  # n_samples,1
        self.n_samples = xy.shape[0]

        self.transform = transform



    def __getitem__(self,index):
        # dataset [0]

        sample = self.x[index],self.y[index]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        # len(dataset)

        return self.n_samples
    
class ToTensor:

    def __call__(self,sample):

        inputs, targets = sample

        return torch.from_numpy(inputs), torch.from_numpy(targets)
    

class MulTransform:
    
    def __init__(self,factor):

        self.factor = factor

    def __call__(self,sample):

        inputs, targets = sample

        inputs *= self.factor

        return inputs, targets



""" 

dataiter = iter(dataloader)

data = next(dataiter)

features,labels = data

print(features,labels) 

"""

composed = torchvision.transforms.Compose([ToTensor(),MulTransform(2)])


dataset = WineDataset(transform=composed)

dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True)


# training loop

num_epochs = 100
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
lr = 0.01
model = nn.Linear(13,1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

for epoch in range(num_epochs):

    for i, (inputs,labels) in enumerate(dataloader):

        y_predicted = model(inputs)

        loss = criterion(y_predicted,labels)

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

    
    print(f"Epoch {epoch } loss: {loss:.4f}")

