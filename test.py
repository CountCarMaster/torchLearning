import numpy as np
import torch
import dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda")
GCN_FILTERS = 20
GCN_LAYERS = 2
CAPSULE_DIMENSIONS = 1
ATTENTION_DIMENSIONS = 20
CAPSULE_NUMBER = 8


# 信息聚合
gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

# GCN层的定义
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

# 胶囊初始层的定义
class PrimaryCapsuleLayer(nn.Module):
    def __init__(self, in_units, in_channels, num_units, capsule_dimensions):
        super(PrimaryCapsuleLayer, self).__init__()
        self.num_units = num_units
        self.units = []
        for i in range(self.num_units):
            unit = nn.Conv2d(in_channels=in_channels,
                            out_channels=capsule_dimensions,
                            kernel_size=(in_units, 1),
                            stride=1,
                            bias=True)

            self.add_module("unit_" + str(i), unit)
            self.units.append(unit)

    @staticmethod
    def squash(s):
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        u = [self.units[i](x) for i in range(self.num_units)]
        u = torch.stack(u, dim=1)
        u = u.view(x.size(0), self.num_units, -1)
        return PrimaryCapsuleLayer.squash(u)

# 注意层的定义
class Attention(nn.Module):
    def __init__(self, attention_size_1, attention_size_2):
        super(Attention, self).__init__()
        self.attention_1 = nn.Linear(attention_size_1, attention_size_2)
        self.attention_2 = nn.Linear(attention_size_2, attention_size_1)

    def forward(self, x_in):
        attention_score_base = self.attention_1(x_in)
        attention_score_base = nn.functional.relu(attention_score_base)
        attention_score = self.attention_2(attention_score_base)
        attention_score = nn.functional.softmax(attention_score, dim=0)
        condensed_x = x_in *attention_score
        return condensed_x

# 路由层的定义
class SecondaryCapsuleLayer(torch.nn.Module):
    def __init__(self, in_units, in_channels, num_units, unit_size):
        super(SecondaryCapsuleLayer, self).__init__()
        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.W = torch.nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))

    @staticmethod
    def squash(s):
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1))

        num_iterations = 3

        for _ in range(num_iterations):
            c_ij = torch.nn.functional.softmax(b_ij, dim=2)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = SecondaryCapsuleLayer.squash(s_j)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)
            b_ij = b_ij + u_vj1
            # b_max = torch.max(b_ij, dim = 2, keepdim = True)
            # b_ij = b_ij / b_max.values ## values can be zero so loss would be nan
        return v_j.squeeze(1)


# GCN本体的定义
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.GCNLayer1 = GCNLayer(4, 20)
        self.GCNLayer2 = GCNLayer(20, 20)
        self.FirstCapsule = PrimaryCapsuleLayer(in_units=GCN_FILTERS,
                                                 in_channels=GCN_LAYERS,
                                                 num_units=GCN_LAYERS,
                                                 capsule_dimensions=CAPSULE_DIMENSIONS)
        self.Attention = Attention(GCN_LAYERS * CAPSULE_DIMENSIONS,
                                   ATTENTION_DIMENSIONS)
        self.GraphCapsule = SecondaryCapsuleLayer(GCN_LAYERS,
                                                   CAPSULE_DIMENSIONS,
                                                   CAPSULE_NUMBER,
                                                   CAPSULE_DIMENSIONS)
        self.ClassCapsule = SecondaryCapsuleLayer(CAPSULE_DIMENSIONS,
                                                   CAPSULE_NUMBER,
                                                   1,
                                                   CAPSULE_DIMENSIONS)


    def forward(self, g, features):
        hidden_representations = []

        x = F.relu(self.GCNLayer1(g, features))
        hidden_representations.append(x)
        x = F.relu(self.GCNLayer2(g, x))
        hidden_representations.append(x)

        hidden_representations = torch.cat(tuple(hidden_representations))
        hidden_representations = hidden_representations.view(1, GCN_LAYERS, GCN_FILTERS, -1)
        # print("-----hidden_representations-----")
        # print(hidden_representations.size())
        first_capsule_output = self.FirstCapsule(hidden_representations)
        # first_capsule_output = first_capsule_output.view(-1, GCN_LAYERS * CAPSULE_DIMENSIONS)
        # print("-----first_capsule_output-----")
        # print(first_capsule_output.size())
        # rescaled_capsule_output = self.Attention(first_capsule_output)
        # rescaled_first_capsule_output = rescaled_capsule_output.view(-1, GCN_LAYERS,
        #                                                              CAPSULE_DIMENSIONS)
        # print("-----rescaled_first_capsule_output-----")
        # print(rescaled_capsule_output.size())
        # graph_capsule_output = self.GraphCapsule(rescaled_first_capsule_output)
        # reshaped_graph_capsule_output = graph_capsule_output.view(-1, CAPSULE_DIMENSIONS,
        #                                                           CAPSULE_NUMBER)
        # class_capsule_output = self.ClassCapsule(reshaped_graph_capsule_output)
        # # class_capsule_output = torch.sum(class_capsule_output, dim=1).view(-1, 1)
        # class_capsule_output = class_capsule_output.view(-1, 1)
        # print(first_capsule_output)
        return first_capsule_output.view(-1, 1)



# net即为我们所使用的
net = Net()

# 读取邻接矩阵和特征矩阵（以及labels）
amatrix = np.loadtxt("a-matrix.txt", dtype=int)
features = np.loadtxt("featureMatrix.txt", dtype=float)
labels = np.zeros(3552, dtype=float)
f = open("EB1.txt", "r")
k = 0
for line in f.readlines():
    line = line[:-1]
    a, b = line.split()
    labels[k] = float(b)
    k += 1

# labels的归一化 y_{new} = \frac{y - y_{min}}{y_{max} - y_{min}}
# 反归一化 y = y_{new} * y_{range} + y_{min}
ymax = labels.max()
ymin = 0
labels = labels / ymax

# 构建dgl图
ffrom = np.zeros(2738, dtype=int)
tto = np.zeros(2738, dtype=int)

pin = 0
for i in range(1776) :
    for j in range(1776) :
        if(amatrix[i][j] == 1) :
            ffrom[pin] = i
            tto[pin] = j
            pin += 1

g = dgl.graph((ffrom, tto))

# 添加自环
g.add_edges(g.nodes(), g.nodes())

# 将特征矩阵的numpy数组转化为张量
features = torch.tensor(features)
features = features.to(torch.float32)
label = np.zeros([3552, 1])
for i in range(3552):
    label[i][0] = labels[i]
labels = torch.tensor(label)

# 损失函数和优化器
loss = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

# 评估函数
# def evaluate(model, g, features, labels):
#     model.eval()
#     with torch.no_grad():
#         logits = model(g, features)
#         logits_rsp = logits.cpu().numpy().reshape(-1)
#         labels_rsp = labels.cpu().numpy().reshape(-1)
#         logits_tmp = logits_rsp.argsort()
#         labels_tmp = labels_rsp.argsort()
#         logits_ranks = logits_tmp.argsort()
#         labels_ranks = labels_tmp.argsort()
#         sum = 0
#         for i in range(1776):
#             sum += (labels_ranks[i] - logits_ranks[i]) ** 2
#         return sum / len(labels)

def evaluate(model, g, features, labels):
    model.eval()
    with torch.no_grad():
        sum = 0
        logits = model(g, features)
        for i in range(3552):
            sum += abs(logits[i][0] - labels[i][0])
    return sum

# 转向cuda
# net = net.to(device)
# features = features.to(device)
# labels = labels.to(device)
# g = g.to(device)
# loss = loss.to(device)



# 训练过程
for epoch in range(100) :
    # 训练模式
    net.train()

    # 得到loss
    logits = net(g, features)
    output = loss(logits, labels)

    # 反向传播
    optimizer.zero_grad()
    output.backward()
    optimizer.step()

    # 输出相关参数
    acc = evaluate(net, g, features, labels)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f}".format(
        epoch, output.item(), acc))

# anss = net(g, features)
# print(anss)
# anss = anss.cpu()
# ans = anss.detach().numpy()
# fil = open("compare.txt", "w")
# for i in range(1776) :
#     print("%s %s" % (labels[i][0], ans[i][0]), file=fil)
