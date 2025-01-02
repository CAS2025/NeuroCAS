import torch
import numpy
import torch_geometric
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import scipy.sparse
import pandas
import time

def graph_concat(input, edge_index):
    global output
    graph_index = edge_index
    graph_list = []
    for i in range(input.size(0)):
        graph = torch_geometric.data.Data(x=input[i, :, :], edge_index=graph_index)
        graph_list.append(graph)

    loader = torch_geometric.loader.DataLoader(graph_list, batch_size=input.size(0), shuffle=False)

    for batch in loader:
        output = batch

    return output


class Act_Fun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < 0.4
        return grad_input * temp.float()


act_fun = Act_Fun.apply


def mem_B(fc, alpha, beta, gamma, eta, inputs, spike, mem, hebb):

    thresh = 0.35
    decay = 0.5
    w_decay = 0.9
    state = fc(inputs) + alpha * inputs.bmm(hebb)
    memb = (mem - spike * thresh) * decay + state
    now_spike = act_fun(memb - thresh)
    hebb = w_decay * hebb + torch.bmm(torch.permute(inputs * beta, (0, 2, 1)), ((memb / thresh) - eta).tanh())
    hebb = hebb.clamp(min=-4, max=4)
    return memb, now_spike.float(), hebb


def mem_A(fc, fc3, alpha, beta, gamma, eta, inputs, inputs2, spike, mem, hebb):
    thresh = 0.35
    decay = 0.5
    state = fc(inputs) + fc3(inputs2) + alpha * inputs.bmm(hebb)
    mema = (mem - spike * thresh) * decay + state
    now_spike = act_fun(mema - thresh)
    w_decay = 0.9
    hebb = w_decay * hebb + torch.bmm(torch.permute(inputs * beta, (0, 2, 1)), ((mema / thresh) - eta).tanh())
    hebb = hebb.clamp(min=-4, max=4)
    return mema, now_spike, hebb


class Casual_GLU(torch.nn.Module):

    def __init__(self, in_channels, out_channels, timeblock_padding, layers_dcn, dropout, kernel_size,
                 start_dilation=1):
        super(Casual_GLU, self).__init__()
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.timeblock_padding = timeblock_padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DCN_l1 = DCN(in_dim=in_channels, out_dim=out_channels,
                          kernel_size=kernel_size, layers=layers_dcn, timeblock_padding=timeblock_padding,
                          dilation=start_dilation)
        self.DCN_l2 = DCN(in_dim=in_channels, out_dim=out_channels,
                          kernel_size=kernel_size, layers=layers_dcn, timeblock_padding=timeblock_padding,
                          dilation=start_dilation)
        self.DCN_l3 = DCN(in_dim=in_channels, out_dim=out_channels,
                          kernel_size=kernel_size, layers=layers_dcn, timeblock_padding=timeblock_padding,
                          dilation=start_dilation)

    def forward(self, x):
        res = self.DCN_l3(x)
        out = torch.tanh(self.DCN_l1(x)) * torch.sigmoid(self.DCN_l2(x))
        out = F.dropout(out, self.dropout, training=self.training)
        out = F.relu(out + res)
        return out


class DCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, layers, timeblock_padding, dilation):
        super(DCN, self).__init__()
        self.layers = layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.timeblock_padding = timeblock_padding
        self.filter_convs = torch.nn.ModuleList()

        new_dilation = dilation
        for i in range(layers):
            if i == 0:
                self.filter_convs.append(weight_norm(torch.nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                                                                     kernel_size=(1, kernel_size),
                                                                     dilation=new_dilation)))
            else:
                self.filter_convs.append(weight_norm(torch.nn.Conv2d(in_channels=out_dim, out_channels=out_dim,
                                                                     kernel_size=(1, kernel_size),
                                                                     dilation=new_dilation)))
            new_dilation = new_dilation * 2
        self.receptive_field = numpy.power(2, layers) - 1

    def forward(self, x):
        if self.in_dim == self.out_dim:
            out = x
        else:
            pad_num = self.kernel_size - 1
            out = x
            for i in range(self.layers):
                out = F.pad(out, (pad_num, 0, 0, 0))
                out = self.filter_convs[i](out)
                pad_num = pad_num * 2
        if self.timeblock_padding:
            out = out
        else:
            out = out[:, :, :, self.receptive_field:out.shape[3]]
        return out


def vec2adj_correlation1(position_embed, method, cor_thresh):
    nodes_num = position_embed.size(0)
    batch_size = position_embed.size(1)
    glu_dim = position_embed.size(2)
    embed_dim = position_embed.size(3)
    position_feature = position_embed.reshape(nodes_num, batch_size * glu_dim * embed_dim)
    position_feature = position_feature.permute(1, 0)
    position_graph = position_feature.detach().to('cpu')
    position_graph = position_graph.numpy()
    position_graph = pandas.DataFrame(position_graph)
    matrix = position_graph.corr(method=method)
    matrix = matrix.abs()
    matrix = numpy.array(matrix)
    matrix_tensor = torch.tensor(matrix)
    cor_thresh1 = cor_thresh
    cor_thresh2 = cor_thresh * 1.2

    for i in range(0, 7):
        for j in range(0, 7):
            matrix_tensor[i][j] = torch.where(matrix_tensor[i][j] > cor_thresh1, 1.0, 0.0)

        for j in range(7, 18):
            matrix_tensor[i][j] = torch.where(matrix_tensor[i][j] > cor_thresh2, 1.0, 0.0)

    for i in range(7, 11):
        for j in range(0, 7):
            matrix_tensor[i][j] = torch.where(matrix_tensor[i][j] > cor_thresh2, 1.0, 0.0)

        for j in range(7, 11):
            matrix_tensor[i][j] = torch.where(matrix_tensor[i][j] > cor_thresh1, 1.0, 0.0)

        for j in range(11, 18):
            matrix_tensor[i][j] = torch.where(matrix_tensor[i][j] > cor_thresh2, 1.0, 0.0)

    for i in range(11, 13):
        for j in range(0, 7):
            matrix_tensor[i][j] = torch.where(matrix_tensor[i][j] > cor_thresh2, 1.0, 0.0)

        for j in range(7, 11):
            matrix_tensor[i][j] = torch.where(matrix_tensor[i][j] > cor_thresh2, 1.0, 0.0)

        for j in range(11, 13):
            matrix_tensor[i][j] = torch.where(matrix_tensor[i][j] > cor_thresh1, 1.0, 0.0)

        for j in range(13, 18):
            matrix_tensor[i][j] = torch.where(matrix_tensor[i][j] > cor_thresh2, 1.0, 0.0)

    for i in range(13, 18):
        for j in range(0, 13):
            matrix_tensor[i][j] = torch.where(matrix_tensor[i][j] > cor_thresh2, 1.0, 0.0)

        for j in range(13, 18):
            matrix_tensor[i][j] = torch.where(matrix_tensor[i][j] > cor_thresh1, 1.0, 0.0)

    graph_adj = matrix_tensor.to('cpu')
    torch.Tensor.fill_diagonal_(graph_adj, 0)
    tmp_coo = scipy.sparse.coo_matrix(graph_adj)
    values = tmp_coo.data
    indicate = numpy.vstack((tmp_coo.row, tmp_coo.col))
    u = torch.LongTensor(indicate)
    v = torch.LongTensor(values)
    edge_index = torch.sparse_coo_tensor(u, v, tmp_coo.shape)

    return edge_index, graph_adj


class GLS_network_joint(torch.nn.Module):
    def __init__(self, nodes_num, position_embed_dim, input_dim, seq_length, glu_inputdim, output_dim_glu, dcn_layers,
                 kernel_size, gat_head, graph_corr,
                 gat_outdim, dropout_rate, device_type):
        super(GLS_network_joint, self).__init__()
        self.graph_corr = graph_corr
        self.nodes_num = nodes_num
        self.position_dim = position_embed_dim
        self.input_dim = input_dim
        self.device = device_type
        self.output_dim_glu = output_dim_glu
        self.dcn_layers = dcn_layers
        self.kernel_size = kernel_size
        self.dropout_glu = dropout_rate
        self.input_dim_GLU = glu_inputdim
        self.gat_out_dim = gat_outdim
        self.gat_head = gat_head
        self.seq_len = seq_length

        self.start_conv = weight_norm(torch.nn.Conv2d(in_channels=self.input_dim, out_channels=self.input_dim_GLU,
                                                      kernel_size=(1, 1)))

        self.GLU = Casual_GLU(in_channels=self.input_dim_GLU, out_channels=self.output_dim_glu,
                              layers_dcn=self.dcn_layers,
                              dropout=self.dropout_glu, kernel_size=kernel_size, start_dilation=1, timeblock_padding=True)

        self.gat_conv = torch_geometric.nn.GATv2Conv(in_channels=self.output_dim_glu * self.seq_len,
                                                     out_channels=self.gat_out_dim * self.seq_len,
                                                     heads=self.gat_head, concat=True, dropout_rate=self.dropout_glu)
        self.fc1 = torch.nn.Linear(self.output_dim_glu, 1)
        self.fc2 = torch.nn.Linear(self.seq_len, 1)
        self.fc3 = torch.nn.Linear(self.gat_out_dim * self.gat_head, 1)

        self.fc4 = torch.nn.Linear(self.nodes_num, 1)
        self.fc5 = torch.nn.Linear(self.seq_len * self.gat_out_dim * self.gat_head * self.nodes_num, 1)
        self.activate = torch.nn.ReLU()

    def forward(self, data, position_embed):

        train_data = data
        train_data = train_data.to(self.device).float()
        train_data = torch.unsqueeze(train_data, dim=1)

        x = self.start_conv(train_data)
        embed_vec = torch.zeros(train_data.size(0), self.input_dim_GLU, self.nodes_num, self.position_dim).to(
            self.device)
        if train_data.size(0) == 1:
            x = torch.cat((x, embed_vec), dim=3)

        else:
            x = torch.cat((x, embed_vec), dim=3)

        x = self.GLU(x)

        node_feature = x[:, :, :, :x.shape[3] - self.position_dim]
        position_embed_learned = x[:, :, :, x.shape[3] - self.position_dim:x.shape[3]]
        position_embed_learned = position_embed_learned.permute(0, 2, 3, 1)
        position_embed_learned = self.fc1(position_embed_learned)
        position_embed_learned = position_embed_learned.permute(0, 3, 1, 2)
        position_embed_learned = position_embed_learned.permute(2, 0, 1, 3)
        out = torch.Tensor().to(self.device)

        if self.training:
            edge_index, adj = vec2adj_correlation1(position_embed=position_embed_learned, method='kendall',
                                                   cor_thresh=self.graph_corr)

        else:
            edge_index, adj = vec2adj_correlation1(position_embed=position_embed, method='kendall',
                                                   cor_thresh=self.graph_corr)

        edge_index = edge_index.to(self.device)

        node_feature = node_feature.reshape(node_feature.size(0), self.nodes_num, self.output_dim_glu * self.seq_len)
        graph_data = graph_concat(node_feature, edge_index._indices())
        out = self.gat_conv(graph_data.x, graph_data.edge_index)
        pre = out.view(node_feature.size(0), self.seq_len * self.gat_out_dim * self.gat_head * self.nodes_num)
        pre = self.activate(pre)
        pre = self.fc5(pre)
        pre = pre.float()
        return pre, position_embed_learned, adj


class joint_network(torch.nn.Module):
    def __init__(self, batch_size, seq_length, nodes_num, hebb1_dim, hebb2_dim, glu_inputdim,
                 output_dim_glu, gat_head, device):
        super(joint_network, self).__init__()
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = device
        self.nodes_num = nodes_num
        self.hebb1_dim = hebb1_dim
        self.hebb2_dim = hebb2_dim
        self.glu_inputdim = glu_inputdim
        self.output_dim_glu = output_dim_glu
        self.gat_head = gat_head

        self.alpha1 = torch.nn.Parameter((1e-2 * torch.rand(1)).to(self.device), requires_grad=True)
        self.alpha2 = torch.nn.Parameter((1e-2 * torch.rand(1)).to(self.device), requires_grad=True)

        self.eta1 = torch.nn.Parameter((1e-2 * torch.rand(self.nodes_num, self.hebb1_dim)), requires_grad=True)
        self.eta2 = torch.nn.Parameter((1e-2 * torch.rand(self.nodes_num, self.hebb2_dim)), requires_grad=True)

        self.gamma1 = torch.nn.Parameter((torch.rand(1)).to(self.device), requires_grad=True)
        self.gamma2 = torch.nn.Parameter((torch.rand(1)).to(self.device), requires_grad=True)

        self.beta1 = torch.nn.Parameter((1e-2 * torch.rand(1, self.seq_length)), requires_grad=True)
        self.beta2 = torch.nn.Parameter((1e-2 * torch.rand(1, self.hebb1_dim)), requires_grad=True)

        self.fc1 = torch.nn.Linear(self.seq_length, self.hebb1_dim)
        self.fc2 = torch.nn.Linear(self.hebb1_dim, self.hebb2_dim)
        self.fc3 = torch.nn.Linear(self.hebb2_dim, self.hebb1_dim)
        self.linear = torch.nn.Linear(self.hebb2_dim, self.seq_length)
        self.linear2 = torch.nn.Linear(self.nodes_num, 1)
        self.linear3 = torch.nn.Linear(self.hebb1_dim, 1)
        self.linear4 = torch.nn.Linear(self.nodes_num, 1)
        self.gls = GLS_network_joint(nodes_num=self.nodes_num, position_embed_dim=2, input_dim=1,
                                     seq_length=self.seq_length, glu_inputdim=self.glu_inputdim,
                                     dcn_layers=2, kernel_size=4, graph_corr=0.456,
                                     output_dim_glu=self.output_dim_glu,
                                     gat_head=gat_head,
                                     gat_outdim=1, dropout_rate=0, device_type=self.device)

    def parameter_split(self):
        base_param = []
        for n, p in self.named_parameters():
            if n[:2] == 'fc' or n[:2] == 'fv':
                base_param.append(p)

        local_param = list(set(self.parameters()) - set(base_param))
        return base_param, local_param

    def forward(self, input, hebb1, hebb2, position_embed, wins=10):

        start = time.time()
        h1_mem = h1_spike = h1_sumspike = torch.zeros(input.shape[0], self.nodes_num, self.hebb1_dim,
                                                      device=self.device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(input.shape[0], self.nodes_num, self.hebb2_dim,
                                                      device=self.device)

        inputs2 = torch.zeros(input.shape[0], self.nodes_num, self.hebb2_dim).to(self.device)

        for step in range(wins):
            tau_w = 40
            decay_factor = numpy.exp(- step / tau_w)
            input = input.float()

            h1_mem, h1_spike, hebb1 = mem_A(self.fc1, self.fc3, self.alpha1, self.beta1, self.gamma1,
                                            self.eta1, input * decay_factor, inputs2, h1_spike, h1_mem,
                                            hebb1)
            h1_sumspike = h1_sumspike + h1_spike

            h2_mem, h2_spike, hebb2 = mem_B(self.fc2, self.alpha2, self.beta2, self.gamma2, self.eta2,
                                            h1_spike * decay_factor, h2_spike, h2_mem, hebb2)
            h2_sumspike = h2_sumspike + h2_spike
            inputs2 = h2_spike.to(self.device)

        thresh = 0.35
        outs = h2_mem / thresh
        spike_out = self.linear(outs)
        end = time.time()
        snn_time = (end - start)
        pre, position_embed_learned, adj = self.gls(spike_out, position_embed)

        return pre, position_embed_learned, adj, hebb1.data, hebb2.data, snn_time
