import torch.nn as nn
import torch.nn.functional as F
from pygcn_myself.layers import GraphConvolution

	# nfeat，底层节点的参数，feature的个数
	# nhid，隐层节点个数
	# nclass，最终的分类数
	# dropout参数
	# GCN由两个GraphConvolution层构成
	# 输出为输出层做log_softmax变换的结果

class GCN(nn.Module): # nn.Module类的单继承
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
            # super()函数是用于调用父类(超类)的方法
            # super().__init__()表示子类既能重写__init__()方法又能调用父类的方法
        self.gc1 = GraphConvolution(nfeat,nhid)
        self.gc2 = GraphConvolution(nhid,nclass)
            # 构建第二层GCN，传入的nhid隐藏层的维度为16维，输出的nclass要与最后要判定的类别一致，7个类别
        self.dropout = dropout  # dropout是一个trick， 在训练过程中让某些hid layer节点的权重不工作，防止过拟合的

    def forward(self, x, adj):# 前向传播
        x = F.relu(self.gc1(x, adj))  # gc1(x, adj)进入隐藏层后再放入relu函数中

        x = F.dropout(x, self.dropout, training=self.training)  # trick 防止过拟合
            # 训练神经网络模型时，如果训练样本较少，为了防止模型过拟合，Dropout可以作为一种trick供选择
            # Dropout是指在模型训练时,随机让网络某些隐含层节点的权重不工作，不工作的那些节点可以暂时认为不是网络结构的一部分，
            # 但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了
            # 训练时用dropout(令trainning=True)，评估时关掉dropout(令trainning=False)

        x = self.gc2(x, adj)   # 再经过一层gcn
            # 完成第二层GCN训练后，x.shape是2708*16维的。16是hidden layer的维度，16是传参传进去的
        return F.log_softmax(x, dim=1)
            # 输出为输出层做log_softmax变换的结果，dim表示log_softmax将计算的维度

