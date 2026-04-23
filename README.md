
# 深度学习基础：基于 PyTorch 的 FashionMNIST 图像分类实验报告

## 一、 实验目的
1. 掌握深度学习张量（Tensor）的基本操作与数据管道（DataLoader）的构建方法。
2. 熟练使用 PyTorch 搭建包含非线性激活函数的多层感知机（MLP）分类网络。
3. 学习使用高阶 API（`torchkeras`）简化训练流程，掌握交叉熵损失函数与 Adam 优化器的配置。
4. 掌握模型训练过程中的 Loss 与 Accuracy 可视化方法，并完成模型权重的保存与断点续训机制。

## 二、 实验环境
* **操作系统**：Windows 10 / 11
* **运行平台**：Jupyter Notebook (本地私有化部署)
* **核心环境**：Miniconda + Python 3.9
* **硬件支持**：CPU
* **核心依赖库**：`torch`, `torchvision`, `torchkeras`, `matplotlib`, `torchmetrics`
* **工作目录**：`E:\pytorch` （实现数据与模型缓存的物理级盘符隔离）

## 三、 数据集介绍
本实验采用经典的 **FashionMNIST** 数据集。该数据集包含 70,000 张 `28x28` 像素的灰度图像，分为 10 个类别的服饰（如 T 恤、裤子、套衫、裙子、外套、凉鞋、衬衫、运动鞋、包和短靴）。其中 60,000 张用于训练集，10,000 张用于验证集。

## 四、 实验内容与核心代码

### 4.1 数据预处理与管道构建
使用 `torchvision.transforms.ToTensor()` 将图像转换为 PyTorch 张量，并归一化至 `[0, 1]` 区间。通过 `DataLoader` 将数据按 `batch_size=64` 进行小批量打包，并在训练集上开启 `shuffle=True` 以打乱顺序，提升梯度下降效率与泛化能力。

### 4.2 神经网络架构设计
构建了一个包含一层隐藏层的全连接神经网络（MLP）：
1. **输入层**：通过 `nn.Flatten()` 将 `28x28` 的二维图像展平为长度为 `784` 的一维向量。
2. **隐藏层**：使用 `nn.Linear(784, 256)` 将特征映射至 256 维，并接入 `nn.ReLU()` 激活函数引入非线性拟合能力。
3. **输出层**：使用 `nn.Linear(256, 10)` 将特征降维至 10 维，对应 10 个分类的预测逻辑值（Logits）。

### 4.3 核心实现代码
```python
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchkeras import KerasModel
import torchmetrics
import matplotlib.pyplot as plt

# 1. 数据准备
transform = transforms.Compose([transforms.ToTensor()])
ds_train = torchvision.datasets.FashionMNIST(root="E:/pytorch/data", train=True, download=True, transform=transform)
ds_val = torchvision.datasets.FashionMNIST(root="E:/pytorch/data", train=False, download=True, transform=transform)

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True)
dl_val = torch.utils.data.DataLoader(ds_val, batch_size=64, shuffle=False)

# 2. 模型构建
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10) 
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(self.flatten(x))))

net = Net()

# 3. 模型训练配置
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
model = KerasModel(net, loss_fn=loss_fn, optimizer=optimizer, metrics_dict={"acc": torchmetrics.Accuracy(task='multiclass', num_classes=10)})

# 4. 执行 5 个 Epoch 的训练
dfhistory = model.fit(train_data=dl_train, val_data=dl_val, epochs=5, patience=3, monitor="val_acc", mode="max")
````

## 五、 实验结果与分析

### 5.1 训练过程记录

在设定的 5 个 Epoch 训练过程中，模型表现出稳定的收敛趋势：

- **损失函数（Loss）**：训练集 Loss 与验证集 Loss 均呈现平稳下降趋势，说明模型正在有效学习数据特征，并未出现严重的过拟合现象。
    
- **准确率（Accuracy）**：验证集准确率稳步上升，证明模型对未见过的测试数据具备良好的泛化与分类能力。
    

### 5.2 结果可视化
![[Pasted image 20260410223505.png]]


### 5.3 模型权重保存

基于 `val_acc`（验证集准确率）的监控机制，训练过程中表现最优的模型权重已被自动提取，并持久化保存至本地目录的 `checkpoint.pt` 文件中。后续可直接加载该权重文件进行模型推理与部署。

## 六、 实验总结

本次实验成功配置了纯本地化的深度学习环境，并验证了基于全连接神经网络处理图像分类任务的可行性。通过合理的网络架构设计与 Adam 优化器的引入，模型在 5 个 Epoch 内即收敛至较高的准确率水平。实验全流程加深了对 PyTorch 框架底层机制（前向传播、自动求导、反向传播优化）的理解，圆满完成了本次课程的实验目标。
