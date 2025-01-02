import dill
import torch
import numpy
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas
import scipy.sparse
from sklearn.neighbors import kneighbors_graph




with open('data/test_dataT.pkl', 'rb') as b:  # 测试集数据
    test_dataset = dill.load(b)
with open('data/test5.pkl', 'rb') as a:  # 训练数据
    testid_dataset = dill.load(a)

test = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
testid = DataLoader(dataset=testid_dataset, batch_size=1, shuffle=False)
models = torch.load('training_result/pulse49model.pt')
position_embed = torch.load('training_result/pos_vec_final.pt')
# adj = torch.load('training_result/2adj.pt')


device = 'cuda'
num_nodes = 18
seq_length = 30
loss_function = torch.nn.MSELoss().to(device)

models.eval()
pred_list = []
loss = 0
pre_tensor = torch.Tensor().cpu()
target_tensor = torch.Tensor().cpu()
hebb1_dim = 256
hebb2_dim = 512
hebb1 = torch.zeros(1, seq_length, hebb1_dim, device=device)
hebb2 = torch.zeros(1, hebb1_dim, hebb2_dim, device=device)


for i, (data) in enumerate(testid):
    # 检查 data 张量的第三个维度是否为零
    if data.shape[2] == 0:
        # 维度为零，跳过当前迭代
        continue
    test_data = data[:, 2:20, :].to(device)
    rul = data[:, 23:24, -1]
    # 检查输入形状是否符合预期
    if test_data.shape[2] != seq_length:
        continue  # 跳过该批次
    pre, position_embed_learned, adj, hebb1, hebb2 = models(input=test_data, hebb1=hebb1, hebb2=hebb2,
                                                            position_embed=position_embed)

    pre = pre * 125
    rul = rul * 125

    pre = pre.to('cpu')
    pre_tensor = torch.cat((pre_tensor, pre), dim=0)
    pre_tensor = torch.detach(pre_tensor)
    target_tensor = torch.cat((target_tensor, rul), dim=0)
    target_tensor = torch.detach(target_tensor)
    loss = loss + loss_function(pre.cpu(), rul.cpu())
test_loss = loss_function( target_tensor, pre_tensor)
print('test loss = ' + str(test_loss))

def mape(y_true, y_pred):
    """
    计算 MAPE - 平均绝对百分比误差
    参数:
    y_true (torch.Tensor): 实际值
    y_pred (torch.Tensor): 预测值

    返回:
    float: MAPE值
    """
    mask = y_true != 0
    mape = torch.mean(torch.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


print(mape(target_tensor, pre_tensor))
pre_tensor = pre_tensor.to('cpu')
target_tensor = target_tensor.to('cpu')
result = torch.cat((pre_tensor, target_tensor), dim=1)
result = numpy.array(result)
df = pandas.DataFrame(result)
df.to_excel('result.xlsx', index=False)

p = numpy.array(pre_tensor)
t = numpy.array(target_tensor)


plt.figure(figsize=(10, 4))
plt.plot(pre_tensor, 'r--o', label='Predicted RUL')
plt.plot(target_tensor, 'b--v', label='Actural RUL')
plt.xlabel('Samples')
plt.ylabel('Remaining Useful Life(cycle)')
plt.title('Test Engine ID5')
plt.legend(loc='upper right')
plt.show()