import copy
import dill
import torch
import time
from torch.utils.data import DataLoader
import model
import xlwt
import networkx
import math

with open('data/train_data.pkl', 'rb') as a:  # 训练数据
    train_dataset = dill.load(a)
with open('data/test_data.pkl', 'rb') as b:  # 测试集数据
    test_dataset = dill.load(b)

book = xlwt.Workbook()
sheet = book.add_sheet('sheet1')
sheet.write(0, 0, 'epoch')
sheet.write(0, 1, 'testloss')
sheet.write(0, 2, 'total')
sheet.write(0, 3, 'MAPE Loss')
sheet.write(0, 4, 'Edge number')
sheet.write(0, 5, 'training time')
sheet.write(0, 6, 'Score')

epochs = 80
gsl_epoch = 30
batch_size = 32
seq_length = 30
max_cycle = 125
device = 'cuda'
glu_inputdim = 8
num_nodes = 18
position_embed_dim = 2
glu_output_dim = 16
hebb1_dim = 256
hebb2_dim = 512
gat_head = 6


def mape(y_true, y_pred):

    mask = y_true != 0
    mape = torch.mean(torch.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape


models = model.joint_network(batch_size=batch_size, seq_length=seq_length, nodes_num=num_nodes, hebb1_dim=hebb1_dim,
                             hebb2_dim=hebb2_dim, glu_inputdim=glu_inputdim, output_dim_glu=glu_output_dim,
                             gat_head=gat_head, device='cuda')

position_embed = torch.ones(batch_size, glu_inputdim, num_nodes, position_embed_dim).to(device)
position_embed_final = None

param_base, param_local = models.parameter_split()
models.to(device)
loss_function = torch.nn.MSELoss().cuda()
train = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
optim_base = torch.optim.Adam(param_base, lr=0.0001)
optim_local = torch.optim.Adam(param_local, lr=5e-5)

train_loss = []
test_loss_list = []


for epoch in range(epochs):
    start = time.time()
    models.train()
    hebb1 = torch.zeros(batch_size, seq_length, hebb1_dim, device=device)
    hebb2 = torch.zeros(batch_size, hebb1_dim, hebb2_dim, device=device)
    optim_base.step()
    optim_local.step()
    loss_epoch = 0
    edge_num = 0
    total_snn_time = 0
    total_batches = 0
    for i, (data) in enumerate(train):
        optim_base.zero_grad()
        rul = data[:, -1, -1].to(device).float()
        rul = rul.unsqueeze(-1)
        train_data = data[:, 2:20, :].to(device)
        pre, position_embed_learned, adj, hebb1, hebb2, snn_time = models(input=train_data, hebb1=hebb1, hebb2=hebb2,
                                                                          position_embed=position_embed)

        total_snn_time = total_snn_time + snn_time
        total_batches += train_data.size(0)

        pre = pre * max_cycle
        rul = rul * max_cycle
        loss = loss_function(pre.cpu(), rul.cpu())
        loss_epoch = loss_epoch + copy.deepcopy(float(loss))
        loss.backward(retain_graph=False)
        optim_base.step()

        if epoch < gsl_epoch:
            position_embed = position_embed_learned.detach()

        elif epoch == gsl_epoch:
            position_embed_final = position_embed_learned.detach()
            position_embed = position_embed_final

        else:
            position_embed = position_embed_final

    sheet.write(epoch + 1, 0, epoch)
    sheet.write(epoch + 1, 2, float(loss_epoch))
    torch.save(models, 'training_result/' + 'pulse' + str(epoch) + 'model.pt')

    average_snn_time = total_snn_time / total_batches
    sheet.write(epoch + 1, 5, float(total_snn_time))
    print('epoch snn time = ' + str(total_snn_time))

    if epoch < gsl_epoch:
        torch.save(adj, 'training_result/' + str(epoch) + 'adj.pt')
        torch.save(position_embed, 'training_result/' + str(epoch) + 'pos_vec.pt')
        adj = adj.numpy()
        Graph = networkx.MultiDiGraph(adj)
        edge_num = Graph.number_of_edges() / 2
        print('graph edge number=' + str(edge_num))

    elif epoch == gsl_epoch:
        torch.save(adj, 'training_result/adj_final.pt')
        torch.save(position_embed, 'training_result/pos_vec_final.pt')
        print('%%%%%%%%%%%graph structure optimization completed%%%%%%%%%%%%%%%')

    loss = 0
    pre_tensor = torch.Tensor().cpu()
    target_tensor = torch.Tensor().cpu()
    hebb1 = torch.zeros(1, seq_length, hebb1_dim, device=device)
    hebb2 = torch.zeros(1, hebb1_dim, hebb2_dim, device=device)
    models.eval()
    active = torch.nn.ReLU()
    for i, (data) in enumerate(test):

        if data.shape[2] == 0:
            continue
        test_data = data[:, 2:20, :].to(device)
        rul = data[:, 23:24, -1]
        if test_data.shape[2] != seq_length:
            continue

        pre, position_embed_learned, adj_test, hebb1, hebb2, snn_time1 = models(input=test_data, hebb1=hebb1,
                                                                     hebb2=hebb2, position_embed=position_embed)

        pre = active(pre)
        pre = pre * max_cycle
        rul = rul * max_cycle
        pre = pre.to('cpu')
        pre_tensor = torch.cat((pre_tensor, pre), dim=0)
        pre_tensor = torch.detach(pre_tensor)
        target_tensor = torch.cat((target_tensor, rul), dim=0)
        target_tensor = torch.detach(target_tensor)
        loss = loss + loss_function(pre.cpu(), rul.cpu())
    test_loss = loss_function(target_tensor, pre_tensor)
    test_mape = mape(target_tensor, pre_tensor)

    score = 0
    for i in range(len(target_tensor)):
        if pre_tensor[i] <= target_tensor[i]:
            score = score + math.exp((target_tensor[i] - pre_tensor[i]) / 13) - 1
        else:
            score = score + math.exp((pre_tensor[i] - target_tensor[i]) / 10) - 1

    end_time = time.time()

    print('epoch ' + str(epoch) + 'test MSE loss = ' + str(test_loss))
    print('epoch ' + str(epoch) + 'test MAPE = ' + str(test_mape))
    print('epoch consume time  ' + str(end_time - start))
    print('Score = ' + str(score))

    sheet.write(epoch + 1, 1, float(test_loss))
    sheet.write(epoch + 1, 3, float(test_mape))
    sheet.write(epoch + 1, 4, float(edge_num))
    sheet.write(epoch + 1, 6, float(score))

    book.save('training log.xls')
