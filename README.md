# [](https://gitee.com/wy666666/dl_-conv_model/blob/master/README.md#dl_conv_model)DL_Conv_model

#### [](https://gitee.com/wy666666/dl_-conv_model/blob/master/README.md#%E4%BB%8B%E7%BB%8D)介绍

深度学习中一些经典的卷积神经网络模型的搭建

#### [](https://gitee.com/wy666666/dl_-conv_model/blob/master/README.md#%E5%8D%9A%E5%AE%A2%E5%9C%B0%E5%9D%80)博客地址

# LeNet

using cuda:0 device
项目路径：E:\workspace\PyCharmProject\DL_Conv_model
数据集路径：E:\workspace\PyCharmProject\DL_Conv_model\dataset\flower_data
Using 2 dataloader workers every process
91
使用3306张图片训练，363张图片验证
------------------数据集加载完成--------------
train epoch[1/10] loss:1.205: 100%|██████████| 1653/1653 [00:34<00:00, 48.15it/s]
100%|██████████| 363/363 [00:02<00:00, 168.89it/s]
194.0
[epoch 1] train_loss: 1.269  val_accuracy: 0.534
train epoch[2/10] loss:2.372: 100%|██████████| 1653/1653 [00:33<00:00, 49.94it/s]
100%|██████████| 363/363 [00:02<00:00, 164.54it/s]
217.0
[epoch 2] train_loss: 1.131  val_accuracy: 0.598
train epoch[3/10] loss:1.444: 100%|██████████| 1653/1653 [00:33<00:00, 49.89it/s]
100%|██████████| 363/363 [00:02<00:00, 165.56it/s]
224.0
[epoch 3] train_loss: 1.028  val_accuracy: 0.617
train epoch[4/10] loss:0.761: 100%|██████████| 1653/1653 [00:33<00:00, 49.78it/s]
100%|██████████| 363/363 [00:02<00:00, 160.88it/s]
245.0
[epoch 4] train_loss: 0.973  val_accuracy: 0.675
train epoch[5/10] loss:1.004: 100%|██████████| 1653/1653 [00:33<00:00, 49.96it/s]
100%|██████████| 363/363 [00:02<00:00, 166.93it/s]
231.0
[epoch 5] train_loss: 0.893  val_accuracy: 0.636
train epoch[6/10] loss:0.503: 100%|██████████| 1653/1653 [00:33<00:00, 49.96it/s]
100%|██████████| 363/363 [00:02<00:00, 166.85it/s]
253.0
[epoch 6] train_loss: 0.874  val_accuracy: 0.697
train epoch[7/10] loss:0.963: 100%|██████████| 1653/1653 [00:33<00:00, 49.57it/s]
100%|██████████| 363/363 [00:02<00:00, 164.77it/s]
247.0
[epoch 7] train_loss: 0.852  val_accuracy: 0.680
  0%|          | 0/1653 [00:00<?, ?it/s]
train epoch[8/10] loss:0.927: 100%|██████████| 1653/1653 [00:33<00:00, 49.90it/s]
100%|██████████| 363/363 [00:02<00:00, 164.10it/s]
242.0
[epoch 8] train_loss: 0.832  val_accuracy: 0.667
  0%|          | 0/1653 [00:00<?, ?it/s]
train epoch[9/10] loss:1.587: 100%|██████████| 1653/1653 [00:33<00:00, 49.50it/s]
100%|██████████| 363/363 [00:02<00:00, 164.50it/s]
250.0
[epoch 9] train_loss: 0.811  val_accuracy: 0.689
  0%|          | 0/1653 [00:00<?, ?it/s]
train epoch[10/10] loss:0.132: 100%|██████████| 1653/1653 [00:33<00:00, 49.69it/s]
100%|██████████| 363/363 [00:02<00:00, 164.92it/s]
234.0
[epoch 10] train_loss: 0.804  val_accuracy: 0.645

# AlexNet

using cuda:0 device
项目路径：E:\workspace\PyCharmProject\DL_Conv_model
数据集路径：E:\workspace\PyCharmProject\DL_Conv_model\dataset\flower_data
Using 2 dataloader workers every process
91
使用3306张图片训练，363张图片验证
------------------数据集加载完成--------------
train epoch[1/10] loss:0.848: 100%|██████████| 1653/1653 [00:31<00:00, 51.89it/s]
100%|██████████| 363/363 [00:02<00:00, 177.98it/s]
167.0
[epoch 1] train_loss: 1.383  val_accuracy: 0.460
train epoch[2/10] loss:1.032: 100%|██████████| 1653/1653 [00:29<00:00, 55.21it/s]
100%|██████████| 363/363 [00:02<00:00, 172.31it/s]
196.0
[epoch 2] train_loss: 1.225  val_accuracy: 0.540
train epoch[3/10] loss:0.377: 100%|██████████| 1653/1653 [00:30<00:00, 54.97it/s]
100%|██████████| 363/363 [00:02<00:00, 168.57it/s]
218.0
[epoch 3] train_loss: 1.110  val_accuracy: 0.601
train epoch[4/10] loss:1.590: 100%|██████████| 1653/1653 [00:30<00:00, 54.86it/s]
100%|██████████| 363/363 [00:02<00:00, 171.33it/s]
231.0
[epoch 4] train_loss: 1.085  val_accuracy: 0.636
train epoch[5/10] loss:0.618: 100%|██████████| 1653/1653 [00:30<00:00, 55.05it/s]
100%|██████████| 363/363 [00:02<00:00, 173.10it/s]
229.0
[epoch 5] train_loss: 1.048  val_accuracy: 0.631
train epoch[6/10] loss:1.490: 100%|██████████| 1653/1653 [00:30<00:00, 54.79it/s]
100%|██████████| 363/363 [00:02<00:00, 163.76it/s]
242.0
[epoch 6] train_loss: 1.010  val_accuracy: 0.667
train epoch[7/10] loss:0.221: 100%|██████████| 1653/1653 [00:29<00:00, 55.22it/s]
100%|██████████| 363/363 [00:02<00:00, 169.77it/s]
234.0
[epoch 7] train_loss: 0.977  val_accuracy: 0.645
train epoch[8/10] loss:0.292: 100%|██████████| 1653/1653 [00:29<00:00, 55.25it/s]
100%|██████████| 363/363 [00:02<00:00, 170.51it/s]
258.0
[epoch 8] train_loss: 0.932  val_accuracy: 0.711
train epoch[9/10] loss:0.673: 100%|██████████| 1653/1653 [00:29<00:00, 55.17it/s]
100%|██████████| 363/363 [00:02<00:00, 170.19it/s]
235.0
[epoch 9] train_loss: 0.922  val_accuracy: 0.647
train epoch[10/10] loss:0.718: 100%|██████████| 1653/1653 [00:29<00:00, 55.16it/s]
100%|██████████| 363/363 [00:02<00:00, 167.17it/s]
238.0
[epoch 10] train_loss: 0.915  val_accuracy: 0.656

# VggNet
