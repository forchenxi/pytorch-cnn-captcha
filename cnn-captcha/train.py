import torch
from torch import nn
import torch.nn.functional as f
from torch import optim

import os

# from image_process import ImageProcess


class NeuralNetWork(nn.Module):
    def __init__(self, channel, num_classes):
        """
        :param channel: 输入图片的channel
        :param num_classes: 分类数量
        """
        super(NeuralNetWork, self).__init__()
        # 模型一
        # self.convin = nn.Sequential(
        #     nn.Conv2d(channel, 64, kernel_size=(3, 3), padding=1, bias=False),
        #     nn.BatchNorm2d(64),  # 添加BN层效果并不好
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #     nn.Dropout(0.25)
        # )
        # self.convall = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #     nn.Dropout(0.25)
        # )
        # # 承接卷积层和fc层
        # self.fc1 = nn.Sequential(
        #     nn.Linear(64*5*11, 1024),  # 这个输入值需要计算，根据输入图像的尺寸决定（本次输入图像尺寸为40*90）
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )
        # self.dense1 = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_classes),
        # )
        # self.dense2 = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_classes),
        # )
        # self.dense3 = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, num_classes),
        # )

        # 模型二（效果不如模型一好，但是权重文件只有3.8MB）
        self.conv1 = nn.Conv2d(channel, 32, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1)

        # 全连接层
        self.fc1 = nn.Linear(128*2*5, 512)
        self.fc2 = nn.Linear(512, 128)

        # 虽然每个输出都是经过一个线性层，但是要使用分别单独定义的三个层，否则三个输出永远都是一样的
        self.out_put1 = nn.Linear(128, num_classes)   # 最后的输出取决于有多少个分类（这里是13）
        self.out_put2 = nn.Linear(128, num_classes)
        self.out_put3 = nn.Linear(128, num_classes)
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)

    # n_input是输入图像
    def forward(self, n_input):
        # 模型一
        # # 进行卷积、激活和池化操作
        # feature = self.convin(n_input)
        # feature = self.convall(feature)
        # feature = self.convall(feature)
        #
        # # 对特征层(Tensor类型)进行维度变换，变成两维
        # feature = feature.view(n_input.size(0), -1)  # size(0)是批次大小
        #
        # # 进行全连接操作
        # feature = self.fc1(feature)
        # out_put1 = self.dense1(feature)
        # out_put2 = self.dense2(feature)
        # out_put3 = self.dense3(feature)

        # 模型二
        # 卷积、池化操作
        feature = self.pool(f.relu(self.conv1(n_input)))
        # self.dropout1(feature)
        feature = self.pool(f.relu(self.conv2(feature)))
        # self.dropout1(feature)
        feature = self.pool(f.relu(self.conv3(feature)))
        # self.dropout1(feature)
        feature = self.pool(f.relu(self.conv4(feature)))

        # 对特征层(Tensor类型)进行维度变换，变成两维
        feature = feature.view(n_input.size(0), -1)  # size(0)是批次大小

        # 全连接层
        feature = f.relu(self.fc1(feature))
        self.dropout1(feature)
        feature = f.relu(self.fc2(feature))

        # 特征层经过最后一次全连接层操作，得到最终要分类的结果 并且每个样本有三个输出值
        out_put1 = self.out_put1(feature)
        out_put2 = self.out_put2(feature)
        out_put3 = self.out_put3(feature)
        return [out_put1, out_put2, out_put3]


if __name__ == "__main__":

    net = NeuralNetWork(1, 13)  # channel=1，classes=13
    print(net)
    x = torch.randn(2, 1, 224, 224)
    y = net(x)
    print(y)

    epochs = 300  # 设置训练轮次
    batch_size = 16

    # 训练部分代码
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    # 随机梯度下降优化
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-6)

    ip = ImageProcess()

    val_loss_min = 0  # 保存训练过程中的最小损失（验证）
    for epoch in range(epochs):

        net.train()  # 训练与测试，BN和Dropout有区别
        # 如果没有BN和Dropout,或者只训练不验证，可以不执行该方法
        train_loss = 0.0  # 实时打印当前损失变化情况
        for batch_idx, data in enumerate(ip.train_loader(batch_size=batch_size)):
            inputs, labels = data
            inputs = torch.from_numpy(inputs)  # 从numpy array转成tensor
            labels = torch.from_numpy(labels).long()  # 输入损失函数要求type为long
            optimizer.zero_grad()  # 先将梯度设置为0

            out_puts = net(inputs)  # 前向传播
            # out_puts的shape(n, batch_size, num_classes) 3x16x13 n表示每个样本包含的分类数量
            # 这里因为输出多个值，所以计算损失把多个损失加在一起
            # labels的shape(batch_size, n) 16x3
            loss = (
                    criterion(out_puts[0], labels[:, 0]) +
                    criterion(out_puts[1], labels[:, 1]) +
                    criterion(out_puts[2], labels[:, 2])
            )

            loss.backward()  # 反向传播
            optimizer.step()

            # 查看网络训练状态(损失是计算几批数据的平均损失)
            train_loss += loss.item()

            # 800个训练样本，batch_size=16, 800/16 = 50（一共50批次）
            # 每10批，打印一次损失
            if (batch_idx+1) % 10 == 0:
                print(f'epoch: {epoch+1}, batch_inx: {batch_idx+1} train loss: {train_loss/160}')
                train_loss = 0.0

        state = {
            'net': net.state_dict(),
            'epoch': epoch+1
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if (epoch+1) % 10 == 0:   # 每10轮保存一次权重
            print(f'saving epoch {epoch+1} mode ...')
            torch.save(state, f'./checkpoint/shanghai_epoch_{epoch+1}.pth')  # pth 与 ckpt

        # 验证部分
        net.eval()
        val_loss = 0.0
        for batch_idx, val_data in enumerate(ip.val_loader(16)):
            inputs, labels = val_data
            inputs = torch.from_numpy(inputs)  # 从numpy array转成tensor
            labels = torch.from_numpy(labels).long()  # 输入损失函数要求type为long
            out_puts = net(inputs)

            loss = (
                    criterion(out_puts[0], labels[:, 0]) +
                    criterion(out_puts[1], labels[:, 1]) +
                    criterion(out_puts[2], labels[:, 2])
            )

            val_loss += loss.item()

            # 100个训练样本，batch_size=16, 100/16 = 6（一共7批次）
            # 一轮计算一次平均损失
            if (batch_idx+1) % 7 == 0:
                print(f'epoch: {epoch+1}, batch_inx: {batch_idx+1} val loss: {val_loss/100}')
                if not val_loss_min:
                    val_loss_min = val_loss
                # 正常是每10轮保存一次权重，当发现这一轮验证损失更小时，也会保存一次权重
                elif val_loss_min >= val_loss:
                    val_loss_min = val_loss
                    print(f'saving epoch {epoch+1} mode ...')
                    torch.save(state, f'./checkpoint/shanghai_epoch_{epoch+1}.pth')
            val_loss = 0.0

    print('training task finished')
