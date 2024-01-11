import os
import json

import torch
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from prettytable import PrettyTable

from repvgg import create_RepVGG_A0


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        class_accuracy = []
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            class_acc = round(TP / np.sum(self.matrix[i, :]), 6) if np.sum(self.matrix[i, :]) != 0 else 0.
            class_accuracy.append(class_acc)
            sum_TP += TP
        acc = sum_TP / np.sum(self.matrix)
        print("The model accuracy is ", acc)

        # precision, recall, specificity, and average recall
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "F1", "Accuracy"]
        avg_precision = 0
        avg_recall = 0
        avg_specificity = 0
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 6) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 6) if TP + FN != 0 else 0.
            F1 = round(2 * (Precision * Recall) / (Precision + Recall), 6) if Precision + Recall != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, F1, class_accuracy[i]])
            avg_precision += Precision
            avg_recall += Recall
            avg_specificity += F1
        avg_precision /= self.num_classes
        avg_recall /= self.num_classes
        avg_specificity /= self.num_classes
        table.add_row(
            ["Average", round(avg_precision, 4), round(avg_recall, 4), round(avg_specificity, 4), round(acc, 4)])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=90)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.rcParams['savefig.dpi'] = 1000
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    data_transform = transforms.Compose([#transforms.Resize((224, 224)),  # 随机裁剪
                                         # transforms.RandomHorizontalFlip(),  # 随机反转
                                         # transforms.RandomVerticalFlip(),
                                         transforms.ToTensor(),
        # transforms.Normalize([0.456612, 0.47650352, 0.4028529],
        #                      [0.15292305, 0.13543347, 0.17535591]),
        #  模型标准化出料
  ])

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root+"crops")  # flower data set path
    assert os.path.exists(image_path), "data path {} does not exist.".format(image_path)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val-g50"),
                                            transform=data_transform)

    batch_size = 16
    # 实例化模型
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)
    net = create_RepVGG_A0(num_classes=22)
    # load pretrain weights
    model_weight_path = "D:/VIT/weights/best_model.pth"
    assert os.path.exists(model_weight_path), "cannot find {} file".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    net.to(device)

    # read class_indict
    json_label_path = 'class_indices1.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=22, labels=labels)
    net.eval()
    with torch.no_grad():
        for val_data in tqdm(validate_loader):
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), val_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()

