import math
import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import  Module  # 定义网络层的模块
'''
parameter将一个不可训练的类型tensor转换成可训练的类型parameter，
并将其绑定到这个module里面，所以经过类型转换这个变成了模型的一部分，
成为了模型中根据训练可以改动的参数了。使用这个函数的目的也是为了想
让某些变量在学习过程中不断的修改其值以达到最优解。
'''

class GraphConvolution(Module): # Module类的单继承
    """
    简单的gcn层
    """
    """
        参数：
        in_features:输入特征，每个输入样本的大小
        out_features:输出特征，每个输出样本的大小
        bias:偏置，如果设置为false，则层将不会学习加法偏置。默认值true
        
        属性：
        weight:形状模块的可学习权重
        bias:形状模块的可学习偏差
    """

    def __init__(self, in_features, out_features,bias=True):
        super(GraphConvolution, self).__init__()
            # super函数用于调用父类的方法
            # super().__init__()表示子类既能重写__init__()方法又能调用父类的方法


        self.in_features = in_features  # 初始化
        self.out_features = out_features

            ######################参数定义##########################
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
            # 先转化为张量，再转化为可训练的Parameter对象
            # Parameter用于将参数自动加入到参数列表中
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None) # 第一个参数必须按照字符串形式输入
                # 将parameter对象通过register_parameter进行注册
                # 为模型添加参数

            ######################参数weight bias均匀分布初始化##########################
        self.reset_parameters()


    def reset_parameters(self): # 将参数weight和bias均匀分布初始化
        stdv = 1. / math.sqrt(self.weight.size(1)) #stdv方差
            # size包括(in_features, out_features),size(1)指out_features
            # stdv =1/根号(out_features)
        self.weight.data.uniform_(-stdv, stdv)
            # weight 在区间(-stdv, stdv)之间均匀分布随机初始化，stdv是标准差
        if self.bias is not None: # 变量是否不是none
            self.bias.data.uniform_(-stdv, stdv)  # bias均匀分布初始化

    def forward(self,input ,adj): # 向前传播函数
        support = torch.mm(input, self.weight)
            # input和self.weight矩阵相乘
        output = torch.spmm(adj, support)
            # spmm()是稀疏矩阵乘法，说白了还是乘法而已，只是减小了运算复杂度
            # 最新spmm函数移到了torch.sparse模块下，但是不能用
            # 可以理解为对节点做聚合； 稀疏矩阵的相乘，是之前归一化之后的结果和上一步相乘
        if self.bias is not None:
            return output + self.bias # 返回(系数*输入*权重+偏置)
        else:
            return output             # 返回(系数*输入*权重)


    def __repr__(self): # 打印输出
        return self.__class__.__name__ + '(' \
            + str(self.in_features + '->' \
            + str(self.out_features) + ')')
        # 打印形式 GraphConvolution(输入特征->输出特征)

