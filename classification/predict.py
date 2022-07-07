"""
#-*-coding:utf-8-*- 
# @author: wangyu a beginner programmer, striving to be the strongest.
# @date: 2022/7/4 14:10
"""
import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from LeNet import LeNet

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img_path = input("输入图片路径: ")
    img = Image.open(os.path.join(r"E:\workspace\PyCharmProject\DL_Conv_model\dataset\flower_data\val",img_path))
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img,dim=0)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    model = LeNet().to(device)
    weights_path = "./LeNet.pth"
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
    #                                              predict[predict_cla].numpy())
    print(predict_cla)
    # plt.title(print_res)
    # print(print_res)
    plt.show()


if __name__ == '__main__':
    main()