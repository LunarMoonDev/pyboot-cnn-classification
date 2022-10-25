# importing libraries
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ConvolutionalNetwork

# constants
DATA = './data/'
VERSION = '0.0'

# preping data
transform = transforms.ToTensor()

train_data = datasets.MNIST(root=DATA, train = True, download = True, transform = transform)
test_data = datasets.MNIST(root=DATA, train = False, download = True, transform = transform)

train_loader = DataLoader(train_data, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 10, shuffle = True)

start_time = time.time()

epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# prep the model
torch.manual_seed(42)
model = ConvolutionalNetwork()

# prep the loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# train the model
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # run the training batches first
    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # tally the correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b%600 == 0:
            print(f'epoch: {i:2} batch: {b:4} [{10*b:6}/60000] loss: {loss.item():10.8f} \ acc: {trn_corr.item()*100/(10*b):7.3}%')
        
    train_losses.append(loss.item())
    train_correct.append(trn_corr)

    # run to make prediction with test batches
    # this is so we can track the loss progression of test data
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):

            y_val = model(X_test)

            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()
    
    loss = criterion(y_val, y_test)     # we can put this out since we don't need to train
    test_losses.append(loss.item())
    test_correct.append(tst_corr)

print(f'\nDuration: {time.time() - start_time:.0f} seconds')

# saving the model
torch.save(model.state_dict(), f'./models/model.S.{VERSION}.{int(time.time())}.pt')
pd.DataFrame(np.array(train_losses)).to_csv('./data/processed/train_losses.csv', header = None, index = False)
pd.DataFrame(np.array(test_losses)).to_csv('./data/processed/test_losses.csv', header = None, index = False)
pd.DataFrame(np.array(train_correct)).to_csv('./data/processed/train_correct.csv', header = None, index = False)
pd.DataFrame(np.array(test_correct)).to_csv('./data/processed/test_correct.csv', header = None, index = False)
