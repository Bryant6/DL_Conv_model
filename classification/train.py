"""
#-*-coding:utf-8-*- 
# @author: wangyu a beginner programmer, striving to be the strongest.
# @date: 2022/7/4 10:04
"""
import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from LeNet import LeNet
from AlexNet import AlexNet
from VggNet import VGG
from GoogLeNet import GoogLeNet

def main():
    # 使用GPU
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))

    # 归一化
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    # 加载数据集
    data_root = os.path.abspath(os.path.join(os.getcwd(),"../"))
    print("项目路径：{}".format(data_root))
    image_path = os.path.join(data_root,"dataset","flower_data")
    print("数据集路径：{}".format(image_path))

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path,"train"),transform=data_transform["train"])
    # print(train_dataset)

    # {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
    flower_list = train_dataset.class_to_idx
    # print(flower_list)
    class_dict = dict((key,value) for key,value in flower_list.items())
    # print(class_dict)

    # 写入JSON
    json_str = json.dumps(class_dict,indent=4)
    # print(json_str)
    with open('class_indices.json','w') as json_file:
        json_file.write(json_str)

    batch_size = 1
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=1)
    # print(len(train_dataloader))  # len(train_dataset)/batch_size

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path,"val"),transform=data_transform["val"])
    validate_dataloader = torch.utils.data.DataLoader(validate_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=1)
    print(len(validate_dataloader))
    print("使用{}张图片训练，{}张图片验证".format(len(train_dataset),len(validate_dataset)))
    print("------------------数据集加载完成--------------")

    # 定义模型
    # net = LeNet()
    # net = AlexNet()
    # net = VGG()
    net = GoogLeNet()
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr=0.0002)

    epochs = 10
    # save_path = './LeNet.pth'
    # save_path = './AlexNet.pth'
    save_path = './vgg.pth'
    best_acc = 0.0
    train_steps = len(train_dataloader)

    for epoch in range(epochs):
        # 训练
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_dataloader)

        for step,data in enumerate(train_bar):
            images,labels = data
            optimizer.zero_grad()
            outputs,aux1,aux2 = net(images.to(device))
            loss = loss_function(outputs,labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1,epochs,loss)

        # 验证
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_dataset)
            for val_data in val_bar:
                val_images,val_labels = val_data
                val_images = val_images.unsqueeze(0)
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs,dim=1)[1]
                acc += torch.eq(predict_y,val_labels).sum().item()

        print(acc)
        val_accurate = acc / len(validate_dataset)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # 保存模型
        if val_accurate > best_acc:
            val_accurate = best_acc
            torch.save(net.state_dict(),save_path)

    print('训练完成')


if __name__ == '__main__':
    main()