import os
import cv2
import numpy as np
import random
from os import remove
import math
from torchvision.transforms import transforms
from PIL import Image


"""
数据集分布
{'捌': 206, '减': 346, '肆': 220, '柒': 205, '零': 200, '伍': 214, '加': 358, 
'玖': 189, '壹': 195, '叁': 191, '陆': 206, '乘': 297, '贰': 176}
"""


class ImageProcess:
    channel = 1
    height = 40
    width = 90
    num_classes = 13  # 共13个汉字
    labels_len = 3  # 每个标签包含3个汉字
    words = '0123456789+-x'  # 用字符来代替汉字

    images_path_train = 'D:/captcha/shanghai/train/'
    images_path_val = 'D:/captcha/shanghai/val/'
    images_path_test = 'D:/captcha/shanghai/test/'
    images_train = os.listdir(images_path_train)
    images_val = os.listdir(images_path_val)
    images_test = os.listdir(images_path_test)

    def __init__(self):
        self.x_data_train = None
        self.y_data_train = None
        self.x_data_val = None
        self.y_data_val = None
        self.x_data_test = None
        self.y_data_test = None

        print('预处理图像...')
        self.process_image("train", rotation=True)
        self.process_image("test")
        self.process_image("val")

        print('预处理标签')
        self.process_label("train", rotation=True)
        self.process_label("test")
        self.process_label("val")

        print('处理完成')

    def rename_image(self):
        """
        将文件名中的中文转换为字母和数字
        :return:
        """
        trans_dict = {'零': '0', '壹': '1', '贰': '2', '叁': '3', '肆': '4', '伍': '5',
                      '陆': '6', '柒': '7', '捌': '8', '玖': '9', '加': '+', '减': '-', '乘': 'x'}
        for image in self.images_path_train:
            new_name = image
            for key, value in trans_dict.items():
                new_name = new_name.replace(key, value)
            os.rename(self.images_path_train+image, self.images_path_train+new_name)

    def count_words(self):
        """
         统计数据集各个文字的量是否均衡
        :return:
        """
        word_nums = dict()
        # 每个数据集都看一下
        for image in self.images_test:
            # print(image)
            image_name = image.split("_")[1].replace('.jpg', '')
            for word in image_name:
                if word not in word_nums.keys():
                    word_nums[word] = 1
                else:
                    word_nums[word] += 1
        print(word_nums)

    def group_data(self):
        """
        将数据集划分为训练集、验证集和测试集
        :return:
        """
        # 一开始数据全部在train文件夹下，分别转移一部分到val和test中
        images_list = os.listdir(self.images_path_train)
        for i in range(100):
            t = int(random.random() * len(images_list))
            i_path = images_list[t]
            with open(self.images_path_train + i_path, 'rb') as f1:
                with open(self.images_path_test+i_path, 'wb') as f2:
                    f2.write(f1.read())
                    print('copy {} success'.format(i_path))
            remove(self.images_path_train + i_path)
            del images_list[t]

    def process_label(self, which, rotation=False):
        """
        处理标签
        如果每个样本是单类别，每个类别就一个值，处理成一个长度为batch的列表就可以
        如果每个样本是多类别(假设为n, n>=2)，处理成[batch, n]的二维数组
        :param: which 处理哪个数据集
        :param: rotation 是否旋转（标签这里就是多添加一次，为了总量能与图片对应上）
        :return:
        """
        labels_list = []
        if which == "train":
            images = self.images_train
        elif which == "test":
            images = self.images_test
        else:
            images = self.images_val
        for image in images:
            labels = image.split("_")[1].replace('.jpg', '')
            """
            这部分是ont-hot编码的处理逻辑，在pytorch种实际不需要这样处理，
            这主要取决于 nn.CrossEntropyLoss()的输入参数格式
            参数只需要标签即可, 不需要传one-hot向量
            """
            # 初始化一个 3x13 的矩阵，初始值为0.0
            # result = np.zeros((self.labels_len, self.num_classes), dtype='float32')
            # for i, c in enumerate(labels):
            #     result[i][self.words.index(c)] = 1

            """
            直接处理为 [batch, n]的二维数组 即可
            """
            result = []
            for label in labels:
                result.append(self.words.index(label))
            labels_list.append(result)
            if rotation:
                labels_list.append(result)
        if which == "train":
            self.y_data_train = np.array(labels_list, dtype='int32')
        elif which == "test":
            self.y_data_test = np.array(labels_list, dtype='int32')
        else:
            self.y_data_val = np.array(labels_list, dtype='int32')

    def process_image(self, which, rotation=False):
        """
        处理图片 处理目标 (batch, channel, height, width)
        :param: which 处理哪个数据集
        :param: rotation是否旋转（旋转的话，会另外生成一张图片并且添加到数据集）
        :return:
        """
        images_list = []
        if which == "train":
            images = self.images_train
            images_path = self.images_path_train
        elif which == "test":
            images = self.images_test
            images_path = self.images_path_test
        else:
            images = self.images_val
            images_path = self.images_path_val
        for image in images:
            path = f'{images_path}{image}'
            img = cv2.imread(path)

            # 中值模糊
            # img = cv2.medianBlur(img, 3)
            # 均值模糊
            # img = cv2.blur(img, (2, 2))
            # 高斯模糊
            # img = cv2.GaussianBlur(img, (5, 5), 1)

            # 灰度化
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # 二值化
            ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

            # cv2.namedWindow('captcha', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            # cv2.resizeWindow('captcha', 180, 80)
            # cv2.imshow('captcha', img)
            # cv2.waitKey(0)

            if rotation:
                # 旋转后，会新增一张图片
                rr_img = self.random_rotation(img)
                rr_img = np.array(rr_img, dtype='float32')
                # 归一化
                rr_img /= 255
                images_list.append(np.reshape(rr_img, (1, self.height, self.width)))

            img = np.array(img, dtype='float32')
            # 归一化
            img /= 255
            images_list.append(np.reshape(img, (1, self.height, self.width)))  # 单通道的情况下使用
            # images_list.append(no.transpose(img, (2, 0, 1)))  # 三通道的情况下使用

        if which == "train":
            self.x_data_train = np.array(images_list, dtype='float32')
        elif which == "test":
            self.x_data_test = np.array(images_list, dtype='float32')
        else:
            self.x_data_val = np.array(images_list, dtype='float32')

    def train_loader(self, batch_size=16):
        """
        按批次，将训练数据和标签 迭代返回
        :param batch_size:
        :return:
        """
        batch_nums = math.ceil(len(self.x_data_train)/batch_size)
        for i in range(batch_nums):
            x_train = self.x_data_train[i*batch_size:(i+1)*batch_size]
            y_train = self.y_data_train[i*batch_size:(i+1)*batch_size]
            yield x_train, y_train

    def test_loader(self, batch_size=16):
        """
        按批次，将测试数据和标签 迭代返回
        :param batch_size:
        :return:
        """
        batch_nums = math.ceil(len(self.x_data_test)/batch_size)
        for i in range(batch_nums):
            x_test = self.x_data_test[i*batch_size:(i+1)*batch_size]
            y_test = self.y_data_test[i*batch_size:(i+1)*batch_size]
            yield x_test, y_test

    def val_loader(self, batch_size=16):
        """
        按批次，将验证数据和标签 迭代返回
        :param batch_size:
        :return:
        """
        batch_nums = math.ceil(len(self.x_data_val)/batch_size)
        for i in range(batch_nums):
            x_val = self.x_data_val[i*batch_size:(i+1)*batch_size]
            y_val = self.y_data_val[i*batch_size:(i+1)*batch_size]
            yield x_val, y_val

    # 随机旋转图像
    @staticmethod
    def random_rotation(image):
        image = Image.fromarray(image)  # 传入的image为CV2对象，转换为PIL.Image格式
        # image.show()
        rr = transforms.RandomRotation(degrees=(5, 10))
        rr_image = rr(image)
        # rr_image.show()
        return rr_image  # 返回的依然是PIL.Image格式，但是同样可以直接转为np.array


if __name__ == "__main__":
    ip = ImageProcess()
    ip.count_words()
