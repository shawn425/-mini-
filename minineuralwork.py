# import mmcv
# import torch
# print("torch版本:", torch.__version__)
# print("torch.cuda:", torch.cuda.is_available())
# print("cuda版本:", torch.version.cuda)
# print("torch.cuda.current_device:", torch.cuda.current_device())
# print("cuda.device_count:", torch.cuda.device_count())
# print("mmcv版本:", mmcv.__version__)
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# 设置PyTorch打印选项，使张量输出更加清晰易读
torch.set_printoptions(precision=4, sci_mode=False)

# 随机初始化权重矩阵w和偏置向量b，requires_grad=True表示这些变量需要计算梯度
w = torch.randn(5, 8, requires_grad=True)
b = torch.randn(8, requires_grad=True)

# 创建一个随机输入张量x和一个随机目标张量Y，用于模拟数据
x = torch.randn(1, 5)
Y = torch.randn(1, 8)

# 计算前向传播结果，使用线性组合x@w + b，并通过ReLU激活函数
y = F.relu(x @ w + b)

# 打印前向传播的结果
print(y)

# 计算交叉熵损失，这是分类任务中常用的损失函数
loss = F.cross_entropy(y, Y)

# 打印损失值
print(loss)

# 调用backward()方法进行反向传播，计算权重和偏置的梯度
loss.backward()

# 使用梯度更新权重w，这是一个简单手动实现的梯度下降步骤
w = w - 0.001 * w.grad


# 定义一个简单的卷积神经网络模型
class M_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层卷积层，输入通道数为3（彩色图像），输出通道数为16，卷积核大小为3x3，填充为1保持输出尺寸不变
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 第二层卷积层，输入通道数为16，输出通道数为32，卷积核大小为3x3，填充为1保持输出尺寸不变
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 第一层批归一化层，应用于第一个卷积层的输出，有助于加速训练过程
        self.bn1 = nn.BatchNorm2d(16)
        # 第二层批归一化层，应用于第二个卷积层的输出
        self.bn2 = nn.BatchNorm2d(32)
        # 最大池化层，减少特征图的空间尺寸，通常设置为2x2窗口，步长为2
        self.MaxPool = nn.MaxPool2d(2)
        # 第一个全连接层，输入大小为12*12*32，输出大小为1000
        self.fc1 = nn.Linear(12 * 12 * 32, 1000)
        # 第二个全连接层，输入大小为1000，输出大小为100
        self.fc2 = nn.Linear(1000, 100)
        # 输出层，输入大小为100，输出大小为7，对应7个类别
        self.fc3 = nn.Linear(100, 7)

    def forward(self, x):
        # 第一个卷积层，输出通过ReLU激活函数和批归一化层
        x = F.relu(self.bn1(self.conv1(x)))
        # 第一次最大池化操作，减少特征图的空间尺寸
        x = self.MaxPool(x)
        # 第二个卷积层，输出同样通过ReLU激活函数和批归一化层
        x = F.relu(self.bn2(self.conv2(x)))
        # 第二次最大池化操作
        x = self.MaxPool(x)
        # 将二维特征图展平成一维向量，准备送入全连接层
        x = x.view(-1, 12 * 12 * 32)
        # 第一个全连接层，输出通过ReLU激活函数
        x = F.relu(self.fc1(x))
        # 第二个全连接层，输出同样通过ReLU激活函数
        x = F.relu(self.fc2(x))
        # 输出层，输出也通过ReLU激活函数，但在实际应用中通常使用softmax等其他激活函数
        x = F.relu(self.fc3(x))

        return x


# 数据预处理步骤，包括将图片转换为张量和对数据进行标准化处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

# 加载图像数据集，并应用上述变换，路径为'data/mood'
dataset = ImageFolder('data/mood', transform=transform)
# 创建数据加载器，用于批量加载数据，batch_size设为32，shuffle=True表示每个epoch开始前打乱数据顺序
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 检查是否有可用的GPU设备，如果有则使用GPU，否则使用CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化模型，并将其移动到选定的设备上
model = M_CNN().to(device)

# 定义损失函数，这里使用交叉熵损失，适用于多分类问题
loss_fun = F.cross_entropy

# 定义优化器，使用Adam算法，学习率为0.01，Adam是一种自适应学习率的方法
opt = optim.Adam(model.parameters(), lr=0.01)

# 设置训练轮次
epochs = 8

# 开始训练循环
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        # 获取数据和标签，并将它们移动到选定的设备上
        inputs, labels = data[0].to(device), data[1].to(device)

        # 前向传播，将输入数据送入模型，得到模型的预测结果
        y = model(inputs)

        # 计算损失，将模型的预测结果与真实标签进行比较
        loss = F.cross_entropy(y, labels)

        # 反向传播，计算损失关于模型参数的梯度
        loss.backward()

        # 更新模型参数，优化器根据计算出的梯度调整模型参数
        opt.step()

        # 清零梯度，防止累积影响下次迭代
        opt.zero_grad()

        # 打印每次迭代后的损失值，便于监控训练过程
        print(f"Iteration {i}, Loss: {loss.item()}")

# 注意：在实际应用中，通常还会添加验证和测试步骤，以评估模型的性能
