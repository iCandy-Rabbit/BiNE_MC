import torch


# 保存文件
def ndarray_tostring(array):
    string = ""
    for item in array:
        string += str(item.item()).strip()+" "
    return string+"\n"
# 保存文件
def save_to_file(node_list_u,node_list_v,model_path,userlistIndex,itemlistIndex):
    # "../TrainResult/vectors_u.dat"
    with open(model_path[0],"w") as fw_u:
        idx=0
        for u in node_list_u:
            fw_u.write('u'+str(int(userlistIndex[idx].item()))+" "+ ndarray_tostring(u))
            idx+=1
    # "../TrainResult/vectors_v.dat"
    with open(model_path[1],"w") as fw_v:
        idx=0
        for v in node_list_v:
            fw_v.write('i'+str(int(itemlistIndex[idx].item()))+" "+ndarray_tostring(v))
            idx+=1

# 读取文件
def read_file(uEmd,vEmd):
    u = {}
    udata = []
    with open(uEmd,"r") as fw_u:
        line = fw_u.readline()  
        while line:
            filedata = line.strip().split(" ")
#             userId, uVectors = line.strip().split(" ")# 数据是u0 i0 1  需要映射和更改  更改成 0 0 float(1)  
            uemd = filedata[1:]
            idx = int(float(filedata[0][1:]))
            uVectors = list([ float(i) for i in uemd ])
            if u.get(idx) == None:#这是以字典形式存储
                u[idx] = []
            u[idx]=uVectors
            udata.append(uVectors)
            line = fw_u.readline()  # 读取一行数据
    # 计算得到的是字典，要换成向量的形式 用矩阵的方式最好不过
    vdata = []
    with open(vEmd,"r") as fw_v:
        line = fw_v.readline()
        while line:
            filedata = line.strip().split(" ")
            vemd = filedata[1:]
            idx = int(float(filedata[0][1:]))
            vVectors = list([float(i) for i in vemd])
            vdata.append(vVectors)
            line = fw_v.readline()
    return udata,vdata

