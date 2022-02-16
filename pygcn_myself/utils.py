# 此文件定义了一些需要用的工具函数
import numpy as np
import scipy.sparse as sp   # scipy.sparse稀疏矩阵包
import torch




def encode_onehot(labels): # 将标签转换为onehot形式
    classes = set(labels) # 提取标签类别，并去重操作 形式['NN','RL','Rein'....]
    classes_dict = {c:np.identity(len(classes)) [i,:] for i, c in enumerate(classes)}   # {'RL':array[1,0,0,0,0,0,0],'NN':array[0,1,0,0,0,0,0]....}
    # 这一句主要功能就是进行转化成dict字典数据类型，值为one-hot编码
    # np.identity创建对角矩阵(单位矩阵)，返回主对角线元素为1，其余为0,如np.identity(7) 是7*7的单位矩阵
    # [i,:]就是去上面单位矩阵的第i行
    # get_labels = list(map(classes_dict.get,labels))
    labels_onehot = np.array(list(map(classes_dict.get,labels)),dtype=np.int32)
    # map(function,iterable)是对指定序列iterable中的每一个元素调用function函数
    # 这句话的意思就是获取class_dict的值，将输入一一对应one-hot编码输出
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"): # 加载数据
    """加载引文网络数据集"""
    print("Loading {} dataset....".format(dataset))
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),dtype=np.dtype(str))
        # np.genfromtxt函数用于从.csv文件或.tsv文件中生成数组
        # np.genfromtxt(fname, dtype)
        # frame: 文件名，../data/cora/cora.content  dtype:数据类型，str字符串
    features = sp.csr_matrix(idx_features_labels[:,1:-1],dtype=np.float32)
        # 提取cora的特征，并将其转换成csr矩阵(压缩稀疏行矩阵)
        # [:,1:-1]是指行全部选中，列选取第二列至倒数第二列
        # 这句的功能就是去除论文编号和标签类别，留下每篇论文的词向量，并用稀疏矩阵编码压缩
    labels = encode_onehot(idx_features_labels[:,-1])
        # 提取论文的标签，并转换成one-hot编码形式

    # 构建graph，找出 边和邻接矩阵
    idx = np.array(idx_features_labels[:,0],dtype=np.int32)
        # 提取论文的编号id数组，即第一列
    idx_map = {j:i for i,j in enumerate(idx)}  # idx_map={31336:0,1061127:1,1106406:2...}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path,dataset),dtype=np.int32)
    edges = np.array(list(map(idx_map.get,edges_unordered.flatten()))).reshape(edges_unordered.shape)
        # 节点35和1033之间有边，转化成节点(论文)索引之间有边，[35 1033]->[163 402]，idx_map={35:163}
    adj = sp.coo_matrix((np.ones(edges.shape[0]),(edges[:,0],edges[:,1])),
                        shape=(labels.shape[0],labels.shape[0]),# edges.shape[0]=5429 edges第一列的大小，即边的个数
                        dtype=np.float32)
        # 构建graph的邻接矩阵，coo_matrix构建的是稀疏矩阵，用坐标形式的稀疏矩阵表示，非对称邻接矩阵
        # edges.shape[0]表示引用关系数组的维度数（行数），np.ones全1的n维数组
        # edges[:,0]被引用论文的索引数组作为行号row，edges[:,1]引用论文的索引数组做列号col
        # labels.shape[0]总论文样本的数量，做方阵维数
        # 前面说白了就是 引用论文的索引做列，被引用论文的索引做行，然后在这个矩阵里面填充1，其余填充0
        # print(adj)

    # 构建对称邻接矩阵,计算转置矩阵，将有向图转化为无向图
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # 将非对称邻接矩阵转化成对称邻接矩阵(由有向转化成无向)

    features = normalize(features)
        #因为features基本都是由0组成，只有少部分1，所以压缩稀疏矩阵
    adj = normalize(adj + sp.eye(adj.shape[0]))
        # adj = D-1(A+E)

    # 分割为train，val，test三个集，最终数据加载为torch的格式并且分成三个数据集
    idx_train = range(140) # 0-139为训练集索引
    idx_val = range(200, 500) # 200-499为验证集索引
    idx_test = range(500, 1500) # 500-1499为测试集索引

    features = torch.FloatTensor(np.array(features.todense())) #将特征矩阵转化为张量形式
        # .todense 与csr_matrix对应，将压缩的稀疏矩阵还原
    labels = torch.LongTensor(np.where(labels)[1])
        # np.where(condition),输出满足条件condition（非0）的元素的坐标，np.where()[1]则表示返回列索引的下标值
        # 上句其实就是将每个标签one-hot向量[0,1,0,0,0,0,0]中非0元素的位置输出成标签
    adj = sparse_mx_to_torch_sparse_tensor(adj)
        # 将scipy稀疏矩阵转换为torch稀疏张量

    idx_train = torch.LongTensor(idx_train)     # 训练集索引列表
    idx_val = torch.LongTensor(idx_val)         # 验证集索引列表
    idx_test = torch.LongTensor(idx_test)       # 测试集索引列表
        #转化成张量

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx): # 这里计算D-1A,而不是计算D-1/2AD-1/2,这里的normalize=D-1
    """行标准化稀疏矩阵"""
    # 这个函数思路就是在邻接矩阵基础上转化成出度矩阵，并求D-1A随机游走归一化拉普拉斯算自
    # 函数实现的标准化方法是将输入左乘一个D-1A算子，就是将矩阵每行进行归一化
    rowsum = np.array(mx.sum(1))
        # sum(1)计算输入矩阵的第1维度求和的结果，这里是将二维矩阵的每一行元素求和，即按矩阵行求和
    r_inv = np.power(rowsum,-1).flatten() #r_inv=[0.05 0.0588 0.0454....] 有2708维
    r_inv[np.isinf(r_inv)] = 0.
        # isinf()测试元素是否为正无穷或负无穷，若是返回真，最后返回一个与输入形状相同的布尔数组
        # 如果某一行全为0，则倒数r_inv为无穷大，将这些置为0
        # 这句就是将数组无穷大的元素置为0
    r_mat_inv = sp.diags(r_inv) # 稀疏对角矩阵 有2708*2708维 每一行的形式都是：(0,0) 0.05
        # 构建对角元素为r_inv的对角矩阵
        # sp.diags()根据给定的对象创建对角矩阵，对角线上的元素为给定对象中的元素
    mx = r_mat_inv.dot(mx)  #点乘
        # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行元素都会与对应的r_inv相乘
        # 所谓矩阵点积就是两个矩阵正常相乘而已
    return mx # D-1A

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy稀疏矩阵转换成torch稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
        # tocoo将此矩阵转换成coo格式，astype转换成数组的数据类型
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row,sparse_mx.col)).astype(np.int64))
        # vstack 将两个数组垂直方向堆叠成一个新数组
        # from_numpy是将numpy中的ndarray转化成pytorch中的tensor
        # indices是coo的索引
    values = torch.from_numpy(sparse_mx.data)
        # values是coo的值
    shape = torch.Size(sparse_mx.shape)
        # coo的形状
    return torch.sparse.FloatTensor(indices,values,shape)
        # sparse.FloatTensor构造稀疏张量

def accuracy(output, labels): # 准确率，此函数可参考学习
    preds = output.max(1)[1].type_as(labels)
        # max(1)返回每一行最大值组成的一维数组和索引，output.max(1)[1]表示找出output中最大值所在的索引indice
        # type_as将张量转化成labels类型
        # 例如：output=[0.1,0.2,0.7][0.2,0.6,0.2] preds就是0.7的索引2,和0.6的索引1
        # labels=[0,0,1][0,1,0],索引为2和1  output和label的索引相等，则正确
    correct = preds.eq(labels).double()
        # eq判断preds和labels是否相等，相等的话置1，不相等置0
    correct = correct.sum()
        # 求出相等（置1）的个数
    return correct / len(labels)