# import libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import genfromtxt

# constants
TRAIN_LOSS = './data/processed/train_losses.csv'
TEST_LOSS = './data/processed/test_losses.csv'
TRAIN_CORR = './data/processed/train_correct.csv'
TEST_CORR = './data/processed/test_correct.csv'

# prep the data
train_losses = genfromtxt(TRAIN_LOSS, delimiter = ',', dtype = np.float)
test_losses = genfromtxt(TEST_LOSS, delimiter = ',', dtype = np.float)
train_correct = genfromtxt(TRAIN_CORR, delimiter = ',', dtype = np.float)
test_correct = genfromtxt(TEST_CORR, delimiter = ',', dtype = np.float)

train_losses = torch.tensor(train_losses, dtype = torch.float)
test_losses = torch.tensor(test_losses, dtype = torch.float)
train_correct = torch.tensor(train_correct, dtype = torch.float)
test_correct = torch.tensor(test_correct, dtype = torch.float)

# plotting 'loss at the end of each epoch'
plt.plot(train_losses.tolist(), label = 'training loss')
plt.plot(test_losses.tolist(), label = 'validation loss')
plt.title('Loss at the end of each epoch')
plt.legend()
plt.savefig('./reports/figures/end_loss_epochs.png')

plt.plot([t / 600 for t in train_correct], label = 'training accuracy')
plt.plot([t / 100 for t in test_correct], label = 'training accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend()
plt.savefig('./reports/figures/accuracy_epochs.png')

print([t / 600 for t in train_correct])
print([t / 100 for t in test_correct])
print(len([t / 600 for t in train_correct]))
print(len([t / 100 for t in test_correct]))

