import os
import math
import argparse
from torchvision import transforms, datasets
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision import models
from my_data import MyDataSet
from center_loss import CenterLoss
import torch.nn.utils.prune as prune  #通道剪枝的工具
from shufflenetV2 import shufflenet_v2_x0_5 as create_model
from utils import read_split_data, train_one_epoch, evaluate

import xlwt

book = xlwt.Workbook(encoding='utf-8')  # 创建Workbook，相当于创建Excel
# 创建sheet，Sheet1为表的名字，cell_overwrite_ok为是否覆盖单元格
sheet1 = book.add_sheet(u'Train_data', cell_overwrite_ok=True)
# 向表中添加数据
sheet1.write(0, 0, 'epoch')
sheet1.write(0, 1, 'Train_Loss')
sheet1.write(0, 2, 'Train_Acc')
sheet1.write(0, 3, 'Val_Loss')
sheet1.write(0, 4, 'Val_Acc')
sheet1.write(0, 5, 'lr')
sheet1.write(0, 6, 'Best val Acc')


def main(args):
    best_acc = 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")  # 确定是使用cpu还是GPU

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")   # 权重保存的位置   创建文件weighs

    tb_writer = SummaryWriter()   #模型可视化所用的工具
    # 从这确定了数据集的划分比例 大概是4：1

    data_transform = {       # 数据增强
        "train": transforms.Compose([#transforms.Resize((224, 224)),  # 等比缩小的224
                                     #transforms.RandomHorizontalFlip(p=1),     # 随机反转
                                     # transforms.ColorJitter(brightness=0.9), # 亮度
                                     # transforms.ColorJitter(brightness=1.1),  # 亮度
                                     # transforms.RandomVerticalFlip(p=1),
                                     #ransforms.RandomCrop(200),   #随机裁剪
                                     #transforms.CenterCrop(200),
                                     # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                                     #transforms.RandomRotation(45), # 随机旋转
                                     transforms.ToTensor(),    #  模型标准化出料
                                     #transforms.Normalize([0.456612, 0.47650352, 0.4028529],
                                                         # [0.15292305, 0.13543347, 0.17535591]),
                                     ]),   # 从头训练需要改一下  如果使用预训练模型需要减去某个值
        "val": transforms.Compose([#transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   #transforms.Normalize([0.456612, 0.47650352, 0.4028529],
                                                      #  [0.15292305, 0.13543347, 0.17535591]),
                                   ])}
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "yumi")  # flower data set path
    # 实例化训练数据集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])

    # 实例化验证数据集
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    batch_size = args.batch_size    # batch——size的设置
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=True,
                                                  num_workers=nw)
    model = create_model(num_classes=4).to(device)    # 模型的确定

    images = torch.zeros(1, 3, 224, 224).to(device)  # 要求大小与输入图片的大小一致
    tb_writer.add_graph(model, images, verbose=False)

    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)


    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)   #优化器SGD
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    #lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-9)    # 自定义调整学习率 LambdaLR 学习率调整策略上面确定

    for epoch in range(args.epochs):     #判断训练的次数

        sheet1.write(epoch + 1, 0, epoch + 1)      # 将训练次数写入表格
        sheet1.write(epoch + 1, 5, str(optimizer.state_dict()['param_groups'][0]['lr'])) # 第五个写lr的变化

        # train准确率和损失函数的定义
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)


        sheet1.write(epoch + 1, 1, str(train_loss))     # 将其写入表格
        sheet1.write(epoch + 1, 2, str(train_acc))         # 将其写入表格

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=validate_loader,
                                     device=device,
                                     epoch=epoch)
        scheduler.step()
        sheet1.write(epoch + 1, 3, str(val_loss))        # 将其写入表格
        sheet1.write(epoch + 1, 4, str(val_acc))               # 将其写入表格

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]  #  数据可视化的状态
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_acc:       #  最好的训练权重保存
            best_acc = val_acc
            torch.save(model.state_dict(), "./weights/best_model.pth")
            # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

    sheet1.write(1, 6, str(best_acc))
    book.save('.\Train_data.xlsx')
    print("The Best Acc = : {:.4f}".format(best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)

    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default=r"")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')

    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)