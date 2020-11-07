import torch.nn as nn
import torch
import numpy as np
from common import get_batch,get_train_datas,get_data,ont_hot
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size = 5,
                stride=1,
                padding = 0
            ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size = 5,
                stride=1,
                padding = 2
            ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(32*49*23, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 2)
        )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        res = x.view(x.size(0), -1)
        x = self.dense(res)
        return  x

cnn = CNN()
LR=0.0001
epoch = 20000
# x = torch.from_numpy(np.random.random((32,1,200,97)))

optimizer = torch.optim.Adam(cnn.parameters(), LR)
loss_func = nn.CrossEntropyLoss()
# output = cnn(x.float())

bdata, blabel = get_data("data/badqueries.txt", 0, 23333)
gdata, glabel = get_data("data/goodqueries.txt", 1, 88888)
# datas = bdata+gdata
labels = blabel + glabel
data = bdata + gdata

# print(labels)
for epoch in range(epoch):
    batch, target = get_batch(data, labels, 64)
    # print(torch.Tensor(target).shape)
    input = ont_hot(batch)
    # print(input.shape)
    output = cnn(input)

        #计算误差
    # output.view(x.size(0), -1)
    # print(torch.flatten(output, start_dim=0, end_dim=-1))
    # value = torch.flatten(output, start_dim=0, end_dim=-1)
    value = output
    target = torch.Tensor(target).long()
    print(target)

    loss = loss_func(value, target)
    print(loss)
        #将梯度变为0
    optimizer.zero_grad()
        #反向传播
    loss.backward()
        #优化参数
    optimizer.step()
    if epoch % 5 == 0:
        batch, target = get_batch(data, labels, 32)
        # print(torch.Tensor(target).shape)
        input = ont_hot(batch)
        test_output = cnn(input)
        # squeeze将维度值为1的除去，例如[64, 1, 28, 28]，变为[64, 28, 28]
        pre_y = torch.max(test_output, 1)[1].data.squeeze()
        # 总预测对的数除总数就是对的概率
        c=0
        for i in range(len(pre_y)):
            if pre_y[i]==target[i]:
                c = c+1
        accuracy =  c/ float(len(target))
        print("epoch:", epoch, "| train loss:%.4f" % loss.data, "|test accuracy：%.4f" % accuracy)

torch.save(cnn.state_dict(), '\parameter.pkl')
# print(output.shape)