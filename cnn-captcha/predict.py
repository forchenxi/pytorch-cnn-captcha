import torch
from torch import nn
import cv2
import numpy as np


class NeuralNetWork(nn.Module):
    def __init__(self, channel, num_classes):
        """
        :param channel: 输入图片的channel
        :param num_classes: 分类数量
        """
        super(NeuralNetWork, self).__init__()
        # 模型一
        self.convin = nn.Sequential(
            nn.Conv2d(channel, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),  # 添加BN层
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.convall = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64个卷积核，第一次卷积不池化
            nn.Dropout(0.25),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        self.convout = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        # 承接卷积层和fc层
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 1 * 9, 256),  # 这个输入值需要计算，根据输入图像的尺寸决定（本次输入图像尺寸为30*146）
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.dense1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )
        self.dense3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    # n_input是输入图像
    def forward(self, n_input):
        # 模型一
        # 进行卷积、激活和池化操作
        feature = self.convin(n_input)
        feature = self.convall(feature)

        # 对特征层(Tensor类型)进行维度变换，变成两维
        feature = feature.view(n_input.size(0), -1)  # size(0)是批次大小

        # 进行全连接操作
        feature = self.fc1(feature)
        out_put1 = self.dense1(feature)
        out_put2 = self.dense2(feature)
        out_put3 = self.dense3(feature)

        return [out_put1, out_put2, out_put3]


def predict(img: np.array) -> str:
    img = np.array(img, dtype='float32')
    # 归一化
    img /= 255
    img_np = np.transpose(img, (2, 0, 1))
    img_batch = np.array([img_np], dtype='float32')

    net = NeuralNetWork(3, 14)
    check_point = torch.load('chongqing_ybnsr.pth')   # 加载模型权重
    net.load_state_dict(check_point['net'])
    net.eval()

    images = torch.from_numpy(img_batch)
    out_puts = net(images)

    _, predicted1 = torch.max(out_puts[0], 1)
    _, predicted2 = torch.max(out_puts[1], 1)
    _, predicted3 = torch.max(out_puts[2], 1)
    predict_res = [int(predicted1[0]), int(predicted2[0]), int(predicted3[0])]
    words = '0123456789+-xy'
    predict_res_str = "".join([words[i] for i in predict_res])
    return predict_res_str


if __name__ == "__main__":
    image_processed = cv2.imread('D:/captcha/chongqing_ybnsr/test/1637922000_0x4.jpg')
    # 预测
    result = predict(image_processed)

    print(result)
