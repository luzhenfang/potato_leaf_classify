import torch.cuda
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import LeafDataSet

MODEL_OUTPUT_DIR = r"F:\potato_leaf\models"
LOG_DIR = r"F:\potato_leaf\log"

# cuda
from net import ResNet

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if device == "cuda:0":
    torch.cuda.empty_cache()

# print("running device:" + device)
writer = SummaryWriter(LOG_DIR)

writer.add_text("train_log", "running device:" + device)

# 数据集
train_dataset = LeafDataSet("./dataset/Training/*/*.*")
valid_dataset = LeafDataSet("./dataset/Validation/*/*.*")

# 加载器
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)

# 加载网络

net = ResNet(3)
print(net)

# # 直方图
# for name, param in net.named_parameters():
#     writer.add_histogram(name, param.data.cpu().numpy())

# 查看网络结构
writer.add_graph(net, torch.randn([1, 3, 224, 224]))

net = net.to(device)

# 损失函数 和 优化器
loss_fn = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

total_train_step = 0
total_valid_step = 0


def train(epoch):
    global total_train_step
    net.train()
    # print("第{}轮训练开始".format(epoch))
    # writer.add_text("train_log", "第{}轮训练开始".format(epoch),epoch)

    running_loss = 0.0
    for batch_id, data in enumerate(train_dataloader):
        inputs, targets = data

        inputs, targets = inputs.to(device), targets.to(device)
        writer.add_images("train_img", inputs, total_train_step)

        # 梯度清零
        optimizer.zero_grad()
        outputs = net(inputs)
        # 损失
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_step += 1

        running_loss += loss.item()
        if batch_id % 5 == 0:
            # print("[{},{},{}] loss:{:.2f}".format(epoch, batch_id, len(train_dataloader), running_loss))
            writer.add_text("train_log",
                            "[{},{},{}] loss:{:.2f}".format(epoch, batch_id, len(train_dataloader), running_loss),
                            total_train_step)

            # 记录 loss
            writer.add_scalar("loss", running_loss, total_train_step)
            running_loss = 0.0



def test(epoch):
    net.eval()
    correct = 0
    total = 0
    global total_valid_step
    with torch.no_grad():
        for data in valid_dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            total_valid_step += 1
            writer.add_scalar("accuracy", accuracy, total_valid_step)
            # print("Accuracy on test set:{:.5f}".format(accuracy))
            writer.add_text("test_log", "Accuracy on test set:{:.5f}".format(accuracy), total_valid_step)
            # 高精度模型 30 轮后保存
            if accuracy > 98 and epoch >= 25:
                # 如果出现高精度模型则绘制 保存
                writer.add_scalar("saved model", accuracy, epoch)
                torch.save(net, "{}/model_{}__{:.1f}.pth".format(MODEL_OUTPUT_DIR, epoch, accuracy))
            else:
                writer.add_scalar("epoch accuracy", accuracy, epoch)


if __name__ == '__main__':
    # epochs = 35
    epochs = 80
    for e in tqdm(range(epochs)):
        train(e + 1)
        test(e + 1)
