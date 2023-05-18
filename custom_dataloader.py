import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math


class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt("data/wine.csv",delimiter=",",dtype=np.float32,skiprows=1)

        self.x = torch.from_numpy(xy[:,1:])
        self.y = torch.from_numpy(xy[:,[0]])  # n_samples,1
        self.n_samples = xy.shape[0]

    def __getitem__(self,index):
        # dataset [0]

        return self.x[index],self.y[index]

    def __len__(self):
        # len(dataset)

        return self.n_samples
    

dataset = WineDataset()

dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True)
""" 
dataiter = iter(dataloader)

data = next(dataiter)

features,labels = data

print(features,labels) """


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

