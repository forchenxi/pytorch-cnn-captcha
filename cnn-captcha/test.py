import torch

from train import NeuralNetWork
from image_process import ImageProcess


if __name__ == "__main__":
    # 测试
    net = NeuralNetWork(1, 13)
    net.eval()  # 这个写不写都可以
    check_point = torch.load('./checkpoint/shanghai_epoch_120.pth')
    # check_point = torch.load('shanghai_epoch_80.pth')
    net.load_state_dict(check_point['net'])
    batch_size = 16

    ip = ImageProcess()
    total_image = 0  # 总的图片数量
    correct_image = 0

    total_label = 0  # 总的标签数量
    correct_label = 0
    for data in ip.test_loader(batch_size):
        images, labels = data
        images = torch.from_numpy(images)
        out_puts = net(images)
        # batch_result = []
        _, predicted1 = torch.max(out_puts[0], 1)
        _, predicted2 = torch.max(out_puts[1], 1)
        _, predicted3 = torch.max(out_puts[2], 1)
        # batch_result.append(temp_result)
        for i in range(labels.shape[0]):
            total_image += 1
            total_label += 3
            print(f'true label: {labels[i]}')
            true_label = labels[i]
            print(f'predicted label: {predicted1[i]}  {predicted2[i]}  {predicted3[i]}')
            predicted_label = [int(predicted1[i]), int(predicted2[i]), int(predicted3[i])]
            if list(true_label) == predicted_label:
                correct_image += 1
            if true_label[0] == predicted_label[0]:
                correct_label += 1
            if true_label[1] == predicted_label[1]:
                correct_label += 1
            if true_label[2] == predicted_label[2]:
                correct_label += 1

    print(f'correct_image / total_image: {correct_image}/{total_image}')
    print(f'correct_label / total_label: {correct_label}/{total_label}')
