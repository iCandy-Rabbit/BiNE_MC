from pip import main
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

# 计算f1 map ndcg mrr
def computeResult(item_list,net,test_user,test_item,test_rate):
    node_list_u_,node_list_v_={},{}
    i = 0
    for item in net.u.weight:
        node_list_u_['u'+str(i)] = {}
        node_list_u_['u'+str(i)]['embedding_vectors']= item.cpu().detach().numpy()
        i+=1
    # 对于v 需要在这里映射一下
    i = 0
    for item in net.v.weight:
        node_list_v_['i'+str(item_list[i])] = {}
        node_list_v_['i'+str(item_list[i])]['embedding_vectors']= item.cpu().detach().numpy()
        i+=1
    f1, map, mrr, mndcg = top_N(test_user,test_item,test_rate,node_list_u_,node_list_v_,top_n=10)
    print("f1:",f1,"map:",map,"mrr:",mrr,"mndcg:",mndcg)
    return f1, map, mrr, mndcg



# 计算评价指标
def read_data(filename=None):# 读取数据
#     if filename is None:# 如果文件名为空
#         filename = os.path.join(self.model_path,"ratings_test.dat")
    users,items,rates = set(), set(), {}# 用户、item和评分
    with open(filename, "r", encoding="UTF-8") as fin:
        line = fin.readline()
        while line:
            user, item, rate = line.strip().split()
            if rates.get(user) is None:
                rates[user] = {}
            rates[user][item] = float(rate)# 评分
            users.add(user)
            items.add(item)
            line = fin.readline()
    return users, items, rates

def nDCG(ranked_list, ground_truth):# 评价指标
    dcg = 0
    idcg = IDCG(len(ground_truth))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
            continue
        rank = i+1
        dcg += 1/ math.log(rank+1, 2)
    return dcg / idcg

def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i+2, 2)
    return idcg

def AP(ranked_list, ground_truth):
    hits, sum_precs = 0, 0.0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_truth:
            hits += 1
            sum_precs += hits / (i+1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0.0

def RR(ranked_list, ground_list):

    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            return 1 / (i + 1.0)
    return 0

def precision_and_recall(ranked_list,ground_list):#计算预测 和召回率 ranked_list   ground_list真实列表
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    pre = hits/(1.0 * len(ranked_list))# 命中次数/预测列表长度
    rec = hits/(1.0 * len(ground_list))# 命中次数/真实列表长度
    return pre, rec

#计算topK
def top_N(test_u, test_v, test_rate, node_list_u, node_list_v, top_n):
    recommend_dict = {}
    for u in test_u:
        recommend_dict[u] = {}
        for v in test_v:
            if node_list_u.get(u) is None:
                pre = 0
            else:
                U = np.array(node_list_u[u]['embedding_vectors'])
                if node_list_v.get(v) is None:
                    pre = 0
                else:
                    V = np.array(node_list_v[v]['embedding_vectors'])
                    pre = U.dot(V.T)
                    
            recommend_dict[u][v] = float(pre)
    precision_list = []
    recall_list = []
    ap_list = []
    ndcg_list = []
    rr_list = []
    
    for u in test_u:
        
        tmp_r = sorted(recommend_dict[u].items(),key=lambda x: x[1], reverse=True)[0:min(len(recommend_dict[u]),top_n)]
        tmp_t = sorted(test_rate[u].items(), key=lambda x: x[1], reverse=True)[0:min(len(test_rate[u]),top_n)]
        
        tmp_r_list = []
        tmp_t_list = []
        
        for (item, rate) in tmp_r:# 读取预测topn中的item的名称
            tmp_r_list.append(item)

        for (item, rate) in tmp_t:# 读取测试集中topn中的item的名称
            tmp_t_list.append(item)
        pre, rec = precision_and_recall(tmp_r_list,tmp_t_list)# 计算预测 和召回率
        ap = AP(tmp_r_list,tmp_t_list)
        rr = RR(tmp_r_list,tmp_t_list)
        ndcg = nDCG(tmp_r_list,tmp_t_list)
        precision_list.append(pre)
        recall_list.append(rec)
        ap_list.append(ap)
        rr_list.append(rr)
        ndcg_list.append(ndcg)
        precison = sum(precision_list) / len(precision_list)
        
    recall = sum(recall_list) / len(recall_list)
    #print(precison, recall)
    f1 = 2 * precison * recall / (precison + recall)
    map = sum(ap_list) / len(ap_list)
    mrr = sum(rr_list) / len(rr_list)
    mndcg = sum(ndcg_list) / len(ndcg_list)
    return f1,map,mrr,mndcg
