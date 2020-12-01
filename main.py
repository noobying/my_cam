import argparse
from vgg_gap import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
import datetime


def train(train_loader, model, epoch, criterion, optimizer):
    # 用于训练
    model.train()
    correct, total = 0, 0
    acc_sum, loss_sum = 0, 0
    i = 0
    for batch_idx, (img, label) in tqdm(enumerate(train_loader)):
        img, label = img.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        # calculate accuracy

        correct += (torch.max(output, 1)[1].view(label.size()) == label).sum()
        total += train_loader.batch_size
        train_acc = 100. * correct / total
        acc_sum += train_acc
        i += 1

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tTraining Accuracy: {:.3f}%'.format(
                epoch, batch_idx * len(img), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), train_acc))

    acc_avg = acc_sum / i
    loss_avg = loss_sum / len(train_loader.dataset)
    print('Train Epoch: {}\tAverage Loss: {:.3f}\tAverage Accuracy: {:.3f}%'.format(epoch, loss_avg, acc_avg))
    with open('result/result.txt', 'a') as f:
        f.write('Train Epoch: {}\tAverage Loss: {:.3f}\tAverage Accuracy: {:.3f}%'.format(epoch, loss_avg, acc_avg))



def test(test_loader, model, epoch, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    for img, label in test_loader:
        img, label = img.cuda(), label.cuda()
        output = model(img)
        correct += (torch.max(output, 1)[1].view(label.size()) == label).sum().item()
        test_loss += criterion(output, label).item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    result = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc)

    print(result)
    if epoch % 1 == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S') # 获取字符串形式的时间
        model_name = '{} {} {:.5f}'.format(epoch, time_str, test_acc)
        torch.save(model.state_dict(), './model/1.pth')
        with open('model/savelog.txt', 'a') as f:
            f.write('Save model! model_name:{}\tsave_time:{}\ttest_acc:{:.5f}'.format(model_name, time_str, test_acc))



def main():
    # 使用命令行
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help="批次训练的大小", type=int, default='24')
    parser.add_argument('--learning_rate', help="学习率", type=float, default='0.01')
    parser.add_argument('--epoch', help="训练轮次", type=int, default='4')
    args = parser.parse_args()

    # prepare data
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([  # 不知道为什么要这么做
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    train_data = datasets.ImageFolder('./kaggle/train', transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data = datasets.ImageFolder('./kaggle/test', transform=transform_test)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    net = vgg_cap(num_classes=2)
    net.cuda()
    criterion = nn.CrossEntropyLoss()  # 交叉熵为损失函数
    # print(list(filter(lambda p: p.requires_grad, net.parameters())))
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate,
                                momentum=0.9, weight_decay=5e-4)

    for epoch in range(1, args.epoch):
        train(train_loader, net, epoch, criterion, optimizer)
        test(test_loader, net, epoch, criterion)



if __name__ == '__main__':
    main()
