import argparse
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim # 优化器

from torch.utils.tensorboard import SummaryWriter

from pygcn_myself.utils import load_data, accuracy
from pygcn_myself.models import GCN


# 训练设置
    # argparse 模块可以让人轻松编写用户友好的命令行接口。程序定义它需要的
    # 参数，然后 argparse 将弄清如何从 sys.argv 解析出那些参数。
    # 说白了这个就是自己写一些在命令行的输入的特殊指令完成向程序传入参数并运行。
    # 这里能达到的效果是在命令行启动程序会按照设置的默认参数运行程序，
    # 如果需要更改初始化参数则可以通过命令行语句进行修改。
    # https://docs.python.org/zh-cn/3/library/argparse.html#module-argparse
    # https://blog.csdn.net/lly_zy/article/details/97130496
parser = argparse.ArgumentParser()
    # 使用argparse的第一步是创建一个ArgumentParser对象。
    # ArgumentParser对象包含将命令行解析成Python数据类型所需的全部信息。
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    # 通过调用add_argument()来给一个ArgumentParser添加程序参数信息。
    # 第一个参数 - 选项字符串，用于作为标识
    # action - 当参数在命令行中出现时使用的动作基本类型
    # default - 当参数未在命令行中出现时使用的值
    # type - 命令行参数应当被转换成的类型
    # help - 一个此选项作用的简单描述
    # 此句是 禁用CUDA训练

parser.add_argument('--fastmode', action='store_true', default=False,
                   help='Validate during training pass.')
# 在训练通过期间验证
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
    # 随机种子
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
    # 要训练的epoch数
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate')
    # 最初的学习率
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
    # 权重衰减（参数L2损失）
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
    # 隐藏层单元数量
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
    # dropout率（1-保持概率）

args = parser.parse_args()
    # ArgumentParser通过parse_args()方法解析参数。
    # 这个是使用argparse模块时的必备行，将参数进行关联。

np.random.seed(args.seed)
    # 产生随机种子，以使得结果是确定的
    # https://www.cnblogs.com/lliuye/p/8551660.html
torch.manual_seed(args.seed)
    # 为CPU设置随机种子用于生成随机数，以使得结果是确定的
# if args.cuda:
#     torch.cuda.manual_seed(args.seed)
#     # 为GPU设置随机种子


# 加载数据
adj, features, labels, idx_train, idx_val, idx_test = load_data()
    # 加载数据集并进行初始化处理，返回得到的
    # adj样本关系的对称邻接矩阵的稀疏张量	features样本特征张量	labels样本标签即content中的最后一列，
    # idx_train训练集索引列表	idx_val验证集索引列表	idx_test测试集索引列表

# 模型和优化器
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
    # nfeat输入单元数，shape[1]表示特征矩阵的维度数（列数）
    # nclass输出单元数，即样本标签数=样本标签最大值+1
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,weight_decay=args.weight_decay)
    # 构造一个优化器对象Optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数
    # Adam优化器
    # 获取待优化参数，model.parameters()获取网络的参数，将会打印每一次迭代元素的param而不会打印名字
    # lr学习率	weight_decay权重衰减（L2惩罚）
# if args.cuda:  # 如果使用GUP则执行这里，数据写入cuda，便于后续加速
#     model.cuda() # 模型放GPU上跑
#     features = features.cuda()
#     adj = adj.cuda()
#     labels = labels.cuda()
#     idx_train = idx_train.cuda()
#     idx_val = idx_val.cuda()
#     idx_test = idx_test.cuda()

writer = SummaryWriter("loss_train_logs")

# 定义训练函数
def train(epoch):
    t = time.time()  # 返回当前时间戳
    model.train()
    # 固定语句，主要针对启用BatchNormalization和Dropout
    optimizer.zero_grad()
    # 把梯度置零，也就是把loss关于weight的导数变成0
    output = model(features, adj)
    # 执行GCN中的forward前向传播
    loss_train = F.nll_loss(output[idx_train],labels[idx_train])
    # 最大似然/log似然损失函数，idx_train是140(0~139)
    acc_train = accuracy(output[idx_train],labels[idx_train])
    loss_train.backward()
    optimizer.step()
    # 梯度下降，更新值

    if not args.fastmode:
        # 是否在训练期间进行验证？
        # 单独评估验证集的性能，在验证运行期间停用dropout。
        # 因为nn.functional不像nn模块，在验证运行时不会自动关闭dropout，需要我们自行设置。
        model.eval()
        output = model(features, adj) #前向传播
        # val验证，val是训练过程中的测试集，为了能够边训练边看到训练的结果，及时判断学习状态
        loss_val = F.nll_loss(output[idx_val],labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

        print('Epoch:{:04d}'.format(epoch+1), # 正在迭代的epoch数,epoch从0开始
              'loss_train:{:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),	# 训练集准确率
              'loss_val: {:.4f}'.format(loss_val.item()),	# 验证集损失函数值
              'acc_val: {:.4f}'.format(acc_val.item()),	# 验证集准确率
              'time: {:.4f}s'.format(time.time() - t))	# 运行时间
        writer.add_scalar("loss train",loss_train.item(),epoch)
        # tensorboard --logdir=./pygcn_myself/loss_train_logs --port=6008 注意dir的文件夹位置

# 上面的整个步骤归纳：
# 先将model置为训练状态；梯度清零；
# 将输入送到模型得到输出结果；计算损失与准确率；反向传播求梯度更新参数。

# 定义测试函数
def test(): # 相当于对已有的模型在测试集上运行对应的loss与accuracy
    model.eval()
    output = model(features,adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    # 最大似然/log似然损失函数，idx_test是1000(500~1499)
    acc_test = accuracy(output[idx_test], labels[idx_test])

    print('test set results:',
          'loss={:.4f}'.format(loss_test.item()),# 测试集损失函数值
          'accuracy={:.4f}'.format(acc_test.item()))# 测试集的准确率


# 逐个epoch进行train，最后test
# 训练模型
t_total = time.time()  # 记录当前时间戳

for epoch in range(args.epochs):
    train(epoch) # 训练模型

print("Optimization Finished!") # 优化完成！
print("total time elapsed:{:.4f}s".format(time.time()-t_total)) # 总训练时间

# 测试
test()
writer.close()
"""
这个代码就是做半监督的节点分类。具体说就是，我知道每篇论文
的节点特征，以及论文对应的图网络。现在，我只给部分节点的分
类标签去训练网络，然后预测出每个论文节点的分类标签。
"""
