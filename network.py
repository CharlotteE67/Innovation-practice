import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torch.optim as optim
torch.manual_seed(123)

batch_size = 128
lr = 1e-3
epochs = 75
N = 20

device = torch.device("cuda")

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        # 假设之前数据已经做好了padding
        # [bz,6,N,N]---[bz,6,X,X]
        self.cnn = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=0)
        # self.bn = nn.BatchNorm2d(6)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(6)
        self.bn3 = nn.BatchNorm2d(6)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, stride=1)
        #     ----------------------------------
        self.cnn1 = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(3, stride=1)
        self.cnn2 = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(2, stride=1)
        self.cnn3 = nn.Conv2d(in_channels=6, out_channels=1, kernel_size=2, stride=1, padding=0)
        self.pool3 = nn.MaxPool2d(2, stride=1)

    def forward(self, data):
        result = data
        out = self.cnn(result)
        out = self.bn1(out)
        out = self.relu(out)
        result = self.pool(out)

        out = self.cnn(result)
        out = self.bn2(out)
        out = self.relu(out)
        result = self.pool(out)

        out = self.cnn(result)
        out = self.bn3(out)
        out = self.relu(out)
        result = self.pool(out)

        result1 = self.pool1(self.cnn1(result)).squeeze(1).view(result.shape[0], -1)
        result2 = self.pool2(self.cnn2(result)).squeeze(1).view(result.shape[0], -1)
        result3 = self.pool3(self.cnn3(result)).squeeze(1).view(result.shape[0], -1)
        # [bz,Y*Y]
        result = torch.cat([result1, result2, result3], dim=1)
        return result


class RNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入数据的shape为 [bz,seq_len,1]
        self.rnn = nn.LSTM(1, hidden_size=16, num_layers=2, bidirectional=True, dropout=0.5,
                           batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        out, (h, c) = self.rnn(data)
        # [bz, hid_dim*2]
        hidden = torch.cat([h[-2], h[-1]], dim=1)
        hidden = self.dropout(hidden)
        return hidden


class ConcatLinear(nn.Module):
    # 具体的分类数可以通过output_dim来控制
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 全连接层神经元太多会导致训练变得很慢
        self.model = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, output_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, data):
        data = self.model(data)
        # [bz,9]
        return data


class MultiChannelLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = RNN()
        self.rnn2 = RNN()
        self.rnn3 = RNN()
        self.rnn4 = RNN()
        self.rnn5 = RNN()
        self.rnn6 = RNN()

    def forward(self, data):
        # the data input should be in size [bz,6,400,1]
        result0 = self.rnn1(data[:, 0, :, :])
        result1 = self.rnn2(data[:, 1, :, :])
        result2 = self.rnn3(data[:, 2, :, :])
        result3 = self.rnn4(data[:, 3, :, :])
        result4 = self.rnn5(data[:, 4, :, :])
        result5 = self.rnn6(data[:, 5, :, :])
        # result0-5 of size [bz,32]
        result = torch.cat([result0, result1, result2, result3, result4, result5], dim=1)
        return result


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.multilstm = MultiChannelLSTM()
        self.cnn = CNN()
        # 先写死了
        self.conlinear = ConcatLinear(362, 9)

    def forward(self, data_for_rnn, data_for_cnn):
        rnn_result = self.multilstm(data_for_rnn)
        cnn_result = self.cnn(data_for_cnn)
        result = torch.cat([rnn_result, cnn_result], dim=1)
        result = self.conlinear(result)
        # [bz,9]
        return result


class MyDataSet(Dataset):

    def __init__(self, filepath_1, filepath_2, filepath_3, filepath_4, filepath_5, filepath_6):
        self.data1 = np.loadtxt(filepath_1, delimiter=",")
        self.data2 = np.loadtxt(filepath_2, delimiter=",")
        self.data3 = np.loadtxt(filepath_3, delimiter=",")
        self.data4 = np.loadtxt(filepath_4, delimiter=",")
        self.data5 = np.loadtxt(filepath_5, delimiter=",")
        self.data6 = np.loadtxt(filepath_6, delimiter=",")
        # 计算 length
        self.data_len = len(self.data1)

    def __getitem__(self, index):
        label = torch.from_numpy(np.array(self.data1[index][-1]))
        data1 = torch.from_numpy(self.data1[index][:-1]).view(N * N, 1)
        data2 = torch.from_numpy(self.data2[index][:-1]).view(N * N, 1)
        data3 = torch.from_numpy(self.data3[index][:-1]).view(N * N, 1)
        data4 = torch.from_numpy(self.data4[index][:-1]).view(N * N, 1)
        data5 = torch.from_numpy(self.data5[index][:-1]).view(N * N, 1)
        data6 = torch.from_numpy(self.data6[index][:-1]).view(N * N, 1)
        # [6,N*N=400,1]
        data = torch.stack([data1, data2, data3, data4, data5, data6], dim=0)
        return data, label

    def __len__(self):
        return self.data_len



train_db = MyDataSet("laX.csv", "laY.csv", "laZ.csv", "gyX.csv", "gyY.csv", "gyZ.csv")
train_db, val_db,test_db = torch.utils.data.random_split(train_db, [int(len(train_db) * 0.8), len(train_db)-int(len(train_db) * 0.8)-10, 10])
train_loader = torch.utils.data.DataLoader(train_db, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_db, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_db, batch_size=batch_size, shuffle=True)

net = Net().to(device)
optimizer = optim.Adam(net.parameters(),lr=lr)

criteon = nn.CrossEntropyLoss().to(device)


def train(net, iterator, optimizer, criteon):
    net.train()
    avg_correct = []
    for batch_idx, (data, label) in enumerate(iterator):
        rnn_data = data.float().to(device)
        cnn_data = data.float().view(-1, 6, N, N).to(device)
        label = label.long().to(device)
        # 预测值[bz,9]
        logits = net(rnn_data, cnn_data)
        loss = criteon(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算正确率
        pred = logits.data.max(1)[1]
        correct = pred.eq(label.data).float()
        acc = correct.sum()/len(correct)
        avg_correct.append(acc)

        if batch_idx % 10 == 0:
            print("====================================================")
            print("At batch %d loss = %f" % (batch_idx, loss.item()))
            print("At batch %d acc = %f" % (batch_idx, acc))

    avg_acc = torch.tensor(avg_correct).mean()
    print('avg acc:', avg_acc)
    print("=====================================================")


def val(net, iterator, criteon):
    net.eval()
    val_loss = []
    avg_correct = []
    for batch_idx, (data, label) in enumerate(iterator):
        rnn_data = data.float().to(device)
        cnn_data = data.float().view(-1, 6, N, N).to(device)
        label = label.long().to(device)
        # 预测值[bz,9]
        logits = net(rnn_data, cnn_data)
        loss = criteon(logits, label).item()
        val_loss.append(loss)
        # 计算正确率
        pred = logits.data.max(1)[1]
        correct = pred.eq(label.data).float()
        acc = correct.sum()/len(correct)
        avg_correct.append(acc)

    print("validation loss : ", val_loss)
    avg_acc = torch.tensor(avg_correct).mean()
    print('validation avg acc:', avg_acc)

def test(net, iterator):
    net.eval()
    avg_correct = []

    for batch_idx, (data, label) in enumerate(iterator):
        rnn_data = data.float().to(device)
        cnn_data = data.float().view(-1, 6, N, N).to(device)
        label = label.long().to(device)

        # 预测值[bz,9]
        logits = net(rnn_data, cnn_data)
        # 计算正确率
        pred = logits.data.max(1)[1]
        for i in pred:
          if i.item()==0 or i.item()==1 or i.item()==2:
            print("交叉")
          if i.item()==3 or i.item()==4 or i.item()==5:
            print("圆圈")
          if i.item()==6 or i.item()==7 or i.item()==8:
            print("前冲")
        correct = pred.eq(label.data).float()
        acc = correct.sum()/len(correct)
        avg_correct.append(acc)

    avg_acc = torch.tensor(avg_correct).mean()
    print('test avg acc:', avg_acc)


for epoch in range(epochs):
    train(net, train_loader, optimizer, criteon)
    val(net, val_loader, criteon)
print("===================开始预测==========================")
test(net,test_loader)
torch.save(net.state_dict(),"model")
