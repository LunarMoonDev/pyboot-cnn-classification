# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from model import ConvolutionalNetwork

# constants
DATA = './data/'
MODEL_NAME = 'model.S.0.0.1666439593.pt'

# preping data
transform = transforms.ToTensor()
test_data = datasets.MNIST(root = DATA, train = False, download = True, transform = transform)
test_load_all = DataLoader(test_data, batch_size = 10000, shuffle = False)

# preping the model
torch.manual_seed(4)
model = ConvolutionalNetwork()
model.load_state_dict(torch.load(f'./models/{MODEL_NAME}'))
model.eval()

# prep loss func
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()

print(f'Test accuracy: {correct.item()}/{len(test_data)} = {correct.item() * 100 / (len(test_data)): 7.3f}%')

# analysis
np.set_printoptions(formatter = dict(int = lambda x: f'{x:4}'))
print(np.arange(10).reshape(1,10))
print()

# confusion matrix
print(confusion_matrix(predicted.view(-1), y_test.view(-1)), "\n")

misses = np.array([])   # extracting the misses
for i in range(len(predicted.view(-1))):
    if predicted[i] != y_test[i]:
        misses = np.append(misses, i).astype('int64')

print("Number of missed classifications: ",len(misses))
print("Head of misses list: ", misses[:10])

r = 12
row = iter(np.array_split(misses, len(misses) // r + 1))
nextrow = next(row)
print("Index: ", nextrow)
print("Label: ", y_test.index_select(0, torch.tensor(nextrow)).numpy())
print("Guess: ", predicted.index_select(0, torch.tensor(nextrow)).numpy())


images = X_test.index_select(0, torch.tensor(nextrow))
im = make_grid(images, nrow = r)
plt.figure(figsize = (10, 4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
plt.savefig('./reports/figures/missed_classes.png')
