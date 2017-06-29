import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pylab
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
trainset = torchvision.datasets.MNIST(root='./mnistdata', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.MNIST(root='./mnistdata', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

dtype = torch.cuda.FloatTensor


class CJNet(nn.Module):
    def __init__(self):
        super(CJNet, self).__init__()

        self.input_dim = 120
        self.hidden_dim = 50
        self.output_dim = 10

        self.wi_0 = Variable(torch.randn(self.input_dim, self.hidden_dim).type(dtype), requires_grad=True)
        self.wi_1 = Variable(torch.randn(self.input_dim, self.hidden_dim).type(dtype), requires_grad=True)
        self.wi_2 = Variable(torch.randn(self.input_dim, self.hidden_dim).type(dtype), requires_grad=True)
        self.wi_3 = Variable(torch.randn(self.input_dim, self.hidden_dim).type(dtype), requires_grad=True)
        self.w0_1 = Variable(torch.randn(self.hidden_dim, self.hidden_dim).type(dtype), requires_grad=True)
        self.w1_2 = Variable(torch.randn(self.hidden_dim, self.hidden_dim).type(dtype), requires_grad=True)
        self.w2_3 = Variable(torch.randn(self.hidden_dim, self.hidden_dim).type(dtype), requires_grad=True)
        self.w1_o = Variable(torch.randn(self.hidden_dim, self.output_dim).type(dtype), requires_grad=True)
        self.w2_o = Variable(torch.randn(self.hidden_dim, self.output_dim).type(dtype), requires_grad=True)
        self.w3_o = Variable(torch.randn(self.hidden_dim, self.output_dim).type(dtype), requires_grad=True)
        self.parms = [self.wi_0
                      ,self.wi_1
                      ,self.wi_2
                      ,self.wi_3
                      ,self.w0_1
                      ,self.w1_2
                      ,self.w2_3
                      ,self.w1_o
                      ,self.w2_o
                      ,self.w3_o
                      ]

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, self.input_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))

        s0 = F.tanh(x.mm(self.wi_0))
        s1 = x.mm(self.wi_1) + F.tanh(s0.mm(self.w0_1))
        s2 = x.mm(self.wi_2) + F.tanh(s1.mm(self.w1_2))
        s3 = x.mm(self.wi_3) + F.tanh(s2.mm(self.w2_3))
        y = s1.mm(self.w1_o) + s2.mm(self.w2_o) + s3.mm(self.w3_o)

        return y

    # def update_cjp(self):
    #     for p in self.parms:
    #         p.data.sub_(p.grad.data * 0.001)
    #         p.grad.data.zero_()


net = CJNet().cuda()
criterion = nn.CrossEntropyLoss()
parms = list(net.parameters())+net.parms
optimizer = optim.SGD(parms, lr=0.002, momentum=0.9)

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs= net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # net.update_cjp()

        # loss_raw = loss.data[0]
        # s2_no_grad = s2.detach()
        # a = 0.01 / (pow(loss_raw, 2) + 1)
        # net.state = net.state * (1 - a) + s2_no_grad * a

        # print statistics
        running_loss += loss.data[0]
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0


correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs= net(Variable(images.cuda()))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.cuda()).sum()

print('Accuracy of the network on the 10000 test images: %f %%' % (
    100.0 * correct / total))

print('finish')