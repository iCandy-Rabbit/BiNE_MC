from ast import If
from pyexpat import model
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import torch
from torch import nn
import math
import torch.optim as optim
import torch.nn.functional as F
from sklearn import preprocessing
import numpy as np
import operator
import heapq
import time
from torch.autograd import Variable
# 加大学习率 降低训练次数
#引用其他两个文件
import evaluating_indicator
import utils
# import net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#[1,2,3,4,5,6]
#分别对应[2,4,6,8,10,12]
# print("阶数为3，lr=0.9")
l_in = 2
print("读取数据")
# 读取数据
# orgin_data=[]

def read_dataset(filename):
    orgin_data=[]
    with open(filename, encoding="UTF-8") as fin:  # 取出训练数据,并切分为用户，物品，评分 数据
        line = fin.readline()  
        while line:
            user, item, rating = line.strip().split("\t")# 数据是u0 i0 1  需要映射和更改  更改成 0 0 float(1)  
            orgin_data.append((int(user), int(item), float(rating)))
            line = fin.readline()  # 读取一行数据
    return orgin_data

def computeResult():
    node_list_u_,node_list_v_={},{}
    i = 0
    for item in net.u.weight:
        node_list_u_[str(i)] = {}
        node_list_u_[str(i)]['embedding_vectors']= item.cpu().detach().numpy()
        i+=1

    # 对于v 需要在这里映射一下
    i = 0
    for item in net.v.weight:
        node_list_v_[str(item_list[i])] = {}
        node_list_v_[str(item_list[i])]['embedding_vectors']= item.cpu().detach().numpy()
        i+=1

    f1, map, mrr, mndcg = evaluating_indicator.top_N(test_user,test_item,test_rate,node_list_u_,node_list_v_,top_n=10)
    print("f1:",f1,"map:",map,"mrr:",mrr,"mndcg:",mndcg)


filename = r"train.txt"
orgin_data = read_dataset(filename)

print("构建W矩阵")
data = []
user, item = set(), set()
for u, v, r in orgin_data:
    user.add(u)
    item.add(v)
user_list = list(user)
item_list = list(item)
uLen = len(user_list)
vLen = len(item_list)
W = torch.zeros((uLen,vLen))# 矩阵R
for w in orgin_data:
    u, v, r = w
    item_index = item_list.index(v)
    data.append((u,item_index,r))
    W[u][item_index] = r

vlist = [i for i in range(len(item_list))]# 物品索引集合 # 1177
userlistIndex = torch.Tensor(user_list)
itemlistIndex = torch.Tensor(vlist)
#矩阵RU
print("构建RU矩阵")
uMat = torch.mm(W,W.t())# 用户的高阶交互  即二阶交互
# print(torch.matmul(W,W.t()))
# print(torch.mm(W,W.t()))
normWu = uMat/torch.sum(uMat,1).unsqueeze(-1)

k=l_in
mat = normWu
su  = torch.zeros((uLen,uLen)) + normWu
for i in range(1,k):
    mat = torch.mm(mat,normWu)# 检测有没有深拷贝, 没有深拷贝
    su += mat # 计算l阶


print("构建RV矩阵")
vMat = torch.matmul(W.t(),W)#[1177,1177]
normWv = vMat/torch.sum(vMat,1).unsqueeze(-1)

k=l_in
mat = normWv
sv = torch.zeros((vLen,vLen)) + normWv
for i in range(1,k):
    mat = torch.mm(mat,normWv)
    sv+=mat

test_user, test_item, test_rate = evaluating_indicator.read_data(r"test.txt")# 读取测试集





rank = 128

import os
if os.access("init-vectors_u.dat",os.F_OK) and os.access("init-vectors_v.dat",os.F_OK):
    print("正在加载初始向量")
    #若存在 则使用 
    u_vectors,v_vectors = utils.read_file("init-vectors_u.dat","init-vectors_v.dat")
    len_u,len_v = uLen,vLen
    u_vectors = torch.tensor(u_vectors)
    v_vectors = torch.tensor(v_vectors)
else:
    print("正在初始化初始向量")
    # 若不存在-随机初始化
    len_u,len_v = uLen,vLen
    
    vectors_u = np.random.random([len_u, rank])
    u_vectors = preprocessing.normalize(vectors_u, norm='l2')# 再转成tensor
    u_vectors = torch.from_numpy(u_vectors)

    vectors_v = np.random.random([len_v, rank])
    v_vectors = preprocessing.normalize(vectors_v, norm='l2')# 再转成tensor
    v_vectors = torch.from_numpy(v_vectors)
    print("正在保存文件")
    utils.save_to_file(node_list_u=u_vectors,node_list_v=v_vectors,model_path=["init-vectors_u.dat","init-vectors_v.dat"],userlistIndex=userlistIndex,itemlistIndex=itemlistIndex)
    # 保存下来

"""
print("正在初始化初始向量")


len_u,len_v = 6001,1177

vectors_u = np.random.random([len_u, rank])
u_vectors = preprocessing.normalize(vectors_u, norm='l2')# 再转成tensor
u_vectors = torch.from_numpy(vectors_u)

vectors_v = np.random.random([len_v, rank])
vectors_v = preprocessing.normalize(vectors_v, norm='l2')# 再转成tensor
v_vectors = torch.from_numpy(vectors_v)
# print("正在保存文件")
# utils.save_to_file(node_list_u=u_vectors,node_list_v=v_vectors,model_path=["init-vectors_u.dat","init-vectors_v.dat"],userlistIndex=userlistIndex,itemlistIndex=itemlistIndex)
# 保存下来
"""

# 模型
class Net(nn.Module): 
    def __init__(self, u_len,v_len,vectors_u,vectors_v,w):# u_len用户数量，v_len物品数量，u_vectors用户向量，v_vectors物品向量，w权重矩阵
        super(Net, self).__init__()
        
        self.u = nn.Embedding(u_len, rank)# 也可以导入自己设置的向量 比如把正则化的向量u_vectors导入 [6001,128]
        self.u.weight = torch.nn.Parameter(vectors_u,requires_grad=True)#6001,128
        
        self.v = nn.Embedding(v_len, rank)#[1177,128]
        self.v.weight = torch.nn.Parameter(vectors_v,requires_grad=True)
        
        self.W = w # 矩阵R
        
    def forward(self, R):# R [3,17699]
        #step1 计算一阶图
        # R[0] [17699]
        outputs = torch.mul(self.u(R[0].long().to(device)),self.v(R[1].long().to(device))) # r'
        outputs = torch.sum(outputs, dim=1)#[17699]

        #step2 计算高阶图
        u=torch.mm(self.u(userlistIndex.long().to(device)),self.u(userlistIndex.long().to(device)).t())# 用户x用户 m'
        v=torch.mm(self.v(itemlistIndex.long().to(device)),self.v(itemlistIndex.long().to(device)).t())# 物品x物品 n'
        # print(u)
        positivebatch = F.logsigmoid(outputs)
      
        tu = torch.sigmoid(u)# 先进行一次sigmoid计算
        logp_x_u = F.log_softmax(tu.float().to(device), dim=-1)#[6001, 6001]
        p_y_u = F.softmax(su.float().to(device), dim=-1)#[6001, 6001]
        kl_sum_u = F.kl_div(logp_x_u.float().to(device), p_y_u.float().to(device), reduction='sum') 

        tv = torch.sigmoid(v)
        logp_x_v = F.log_softmax(tv.float().to(device),dim=-1)#[1177, 1177]
        p_y_v = F.softmax(sv.float().to(device),dim=-1)#[1177, 1177]
        kl_sum_v = F.kl_div(logp_x_v.float().to(device), p_y_v.float().to(device), reduction='sum')

        R = torch.mm(self.u(userlistIndex.long().to(device)),self.v(itemlistIndex.long().to(device)).t())
   
        norm_u = u/torch.sum(u,1).unsqueeze(-1)#[6001,6001]
       
        tenW = torch.as_tensor(W.to(device))#[6001,1177]
        # print(norm_u.dtype,tenW.dtype)
        # tenW = tenW.float()
        uM = torch.mm(norm_u.float().to(device),tenW.to(device)) #[6001,1177]

        # 再把tenW换成 点乘的试试看

        # KL(MR||R)
        logp_x_u3 = F.log_softmax(uM.float().to(device), dim=-1)#[6001, 6001]
        p_y_u3 = F.softmax(R.float().to(device), dim=-1)#[6001, 6001]
        uloss3 = F.kl_div(logp_x_u3.float().to(device), p_y_u3.float().to(device), reduction='sum') 
        # print("uloss3:",uloss3)

        # sigv = torch.sigmoid(v)
        norm_v = v/torch.sum(v,1).unsqueeze(-1)
        vM = torch.mm(tenW.float().to(device),norm_v.float().to(device))
        logp_x_v3 = F.log_softmax(vM.float().to(device), dim=-1)#[6001, 6001]
        p_y_v3 = F.softmax(R.float().to(device), dim=-1)#[6001, 6001]
        vloss3 = F.kl_div(logp_x_v3.float().to(device), p_y_v3.float().to(device), reduction='sum') 
     
        labda2 = 1
        labda4 = 0.001
        return -torch.mean(positivebatch) + labda2*kl_sum_u + labda2*kl_sum_v + labda4*uloss3 + labda4*vloss3

       

print("初始化模型")
# net = Net(len_u,len_v,u_vectors,v_vectors,W).to(device)
# criterion = nn.KLDivLoss(size_average=False,reduce=False)
net = Net(len_u,len_v,u_vectors,v_vectors,W).to(device)
# 
optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, nesterov=True)


R = []
du = []# 用户集合
dv = []# 物品集合
dr = []# 评分集合


for u,v,r in data:
    du.append(u)
    dv.append(v)
    dr.append(r)
R.append(du)
R.append(dv)
R.append(dr)
R = torch.Tensor(R)# (17699,17699,17699)


print("开始训练")
steps = 30000
# 迭代次数
R.to(device)
for iter in range(0, steps):
    
    runloss = 0
    optimizer.zero_grad()
    outputs = net(R)      


    if iter % 500==0:
        print(f'index : {iter}, Loss: {outputs.item()}')  
        computeResult()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    outputs.backward()
    
    optimizer.step()
    

print("计算评价指标")
# 计算评价指标
computeResult()
# 保存训练结果

print("保存训练结果")
# utils.save_to_file(node_list_u=net.u.weight,node_list_v=net.v.weight,model_path=["result-u-vec.dat","result-v-vec.dat"],userlistIndex=userlistIndex,itemlistIndex=itemlistIndex)

