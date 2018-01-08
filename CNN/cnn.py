import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init

from Utils.data_utils import load_CIFAR10

import numpy as np

BATCH_SIZE = 256
EPOCH_SIZE = 1000
TEST_SIZE = 500

X_train, Y_train, X_val, Y_val, X_test, Y_test, Class_names = load_CIFAR10()
print('Train data shape ' + str(X_train.shape))
print('Train labels shape ' + str(Y_train.shape))
print('Validation data shape ' + str(X_val.shape))
print('Validataion labels shape ' + str(Y_val.shape))
print('Test data shape ' + str(X_test.shape))
print('Test labels shape ' + str(Y_test.shape))


def cal_accuracy(net, x, y, test_size):
    net.eval() # for "Batch Norm" or "Drop Out"

    correct = 0
    for t in range(x.shape[0] // TEST_SIZE):
        vx = Variable(torch.cuda.FloatTensor(x[t * TEST_SIZE: (t+1) * TEST_SIZE]))
        vx = torch.transpose(vx, 1, 3)
        vy = torch.cuda.LongTensor(y[t * TEST_SIZE: (t+1) * TEST_SIZE])

        outputs = net(vx)
        _, prediction = torch.max(outputs.data, 1)
        correct += torch.eq(prediction, vy).sum()

    accuracy = 100 * correct / x.shape[0]

    net.train()

    return accuracy


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # CONV: in, out, kernel size, stride, padding
        # MAX POOL: kernel size, stride, padding
        # in: 32 * 32 * 3
        self.conv1 = nn.Conv2d(3, 64, 5, 1)
        self.conv2 = nn.Conv2d(64, 64, 5, 1)
        self.maxPool1 = nn.MaxPool2d(3, 2)
        # 11 x 11 x 64
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        self.conv5 = nn.Conv2d(128, 128, 3, 1)
        self.maxPool2 = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(2 * 2 * 128, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

        #self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=0.75)


        init.xavier_uniform(self.conv1.weight)
        init.xavier_uniform(self.conv2.weight)
        init.xavier_uniform(self.conv3.weight)
        init.xavier_uniform(self.conv4.weight)
        init.xavier_uniform(self.conv5.weight)
        init.xavier_uniform(self.fc1.weight)
        init.xavier_uniform(self.fc2.weight)
        init.xavier_uniform(self.fc3.weight)

        init.constant(self.conv1.bias, 0)
        init.constant(self.conv2.bias, 0)
        init.constant(self.conv3.bias, 0)
        init.constant(self.conv4.bias, 0)
        init.constant(self.conv5.bias, 0)
        init.constant(self.fc1.bias, 0)
        init.constant(self.fc2.bias, 0)
        init.constant(self.fc3.bias, 0)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxPool1(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxPool2(x)

        x = x.view(-1, self.num_flat_features(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        logits = x

        return logits


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
net.cuda()

criterion = nn.CrossEntropyLoss() # = log softmax + nll loss
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, eps=1e-3, weight_decay=1e-4) # L2?

for e in range(EPOCH_SIZE):

    total_loss = 0
    for b in range(X_train.shape[0] // BATCH_SIZE):
        x = Variable(torch.cuda.FloatTensor(X_train[b * BATCH_SIZE: (b+1) * BATCH_SIZE]))
        x = torch.transpose(x, 1, 3)
        y = Variable(torch.cuda.LongTensor(Y_train[b * BATCH_SIZE: (b+1) * BATCH_SIZE]))

        y_pred = net(x)

        loss = criterion(y_pred, y)
        total_loss += loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation Evaluation
    if e % 1 == 0:
        train_acc = cal_accuracy(net, X_train, Y_train, 500)
        val_acc = cal_accuracy(net, X_val, Y_val, 500)
        print("Epoch: %d, Total Loss: %f, Test Accuracy: %f, Valid Accuracy: %f" % (e, total_loss, train_acc, val_acc))

acc = cal_accuracy(net, X_test, Y_test, 500)
print("Epoch: %d, Total Loss: %f, Test Accuracy: %f" % (e, total_loss, acc))

