import copy
import time
import dill
import numpy
import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset

start = time.time()


def load_dataset_rul_as_dataframe(name):
    train_df = pd.read_csv(name + '/train.txt', sep="\t", header=None)
    print(train_df.shape)

    train_df.columns = ['id', 'cycle', 's1', 's2', 's3', 's4', 's5',
                        's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                        's18', 's19', 's20', 's21', 's22']

    test_df = pd.read_csv(name + '/test.txt', sep="\t", header=None)
    test_df.columns = ['id', 'cycle', 's1', 's2', 's3', 's4', 's5',
                       's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
                       's18', 's19', 's20', 's21', 's22']

    """Data Labeling - generate column RUL"""
    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    train_df = train_df.merge(rul, on=['id'], how='left')
    train_df['RUL'] = train_df['max'] - train_df['cycle']
    train_df.drop('max', axis=1, inplace=True)
    """generate column max for test data"""
    rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    test_df = test_df.merge(rul, on=['id'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df.drop('max', axis=1, inplace=True)
    cycle_max = 125

    """set max RUL as cycle_max"""
    train_rul = pd.DataFrame(train_df.RUL)
    for i in range(train_rul.shape[0]):
        if int(train_rul.iloc[i]) > cycle_max:
            train_rul.iloc[i] = cycle_max

    train_df['RUL'] = train_rul
    test_rul = pd.DataFrame(test_df.RUL)
    for i in range(test_rul.shape[0]):
        if int(test_rul.iloc[i]) > cycle_max:
            test_rul.iloc[i] = cycle_max
    test_df['RUL'] = test_rul

    """trainData Normalize """
    cols_normalize = train_df.columns.difference(['id', 'cycle'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                                 columns=cols_normalize, index=train_df.index)

    join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
    train_df = join_df.reindex(columns=train_df.columns)

    """testData Normalize """
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                                columns=cols_normalize, index=test_df.index)
    test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    test_df = test_join_df.reindex(columns=test_df.columns)
    test_df = test_df.reset_index(drop=True)

    train_all = train_df
    train_all = train_all[['id', 'cycle', 's2', 's3', 's4', 's15', 's16', 's17', 's18', 's5', 's6', 's7', 's8', 's9', 's10',
                           's11', 's12', 's13', 's14', 's19', 's20', 's21', 's22', 'RUL']]
    test_all = test_df
    test_all = test_all[['id', 'cycle', 's2', 's3', 's4', 's15', 's16', 's17', 's18', 's5', 's6', 's7', 's8', 's9', 's10',
                        's11', 's12', 's13', 's14', 's19', 's20', 's21', 's22', 'RUL']]

    return train_all, test_all


""" slide window resample """


def train_data_resample(train_data):

    window_size = 30
    step_size = 1
    data_sheet = torch.Tensor()
    for i in range(train_data['id'].max()):
        train_data_recurrent = train_data.loc[train_data['id'] == i + 1]
        for j in range(int(((train_data_recurrent.shape[0] - window_size) / step_size) + 1)):
            list_sample = train_data_recurrent[j: j + window_size: 1]
            list_sample = numpy.array(list_sample)
            list_sample = torch.tensor(list_sample)
            list_sample = torch.unsqueeze(list_sample, -1)
            data_sheet = torch.cat([data_sheet, copy.deepcopy(list_sample)], dim=-1)

    train_data_sampled = data_sheet
    return train_data_sampled


def test_data_process(test_data):
    window_size = 30
    step_size = 30
    data_sheet = torch.Tensor()
    for i in range(test_data['id'].max()):
        test_data_recurrent = test_data.loc[test_data['id'] == i + 1]
        for j in range(int(((test_data_recurrent.shape[0] - window_size) / step_size) + 1)):
            list_sample = test_data_recurrent[j * step_size: j * step_size + window_size: 1]
            list_sample = numpy.array(list_sample)
            list_sample = torch.tensor(list_sample)
            list_sample = torch.unsqueeze(list_sample, -1)
            data_sheet = torch.cat([data_sheet, copy.deepcopy(list_sample)], dim=-1)

    test_data_sampled = data_sheet
    return test_data_sampled


class test_Dataset_construct(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class train_Dataset_construct(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor

    def __len__(self):
        a = self.data.size(0)
        return a

    def __getitem__(self, item):

        return self.data[item]


data_all = load_dataset_rul_as_dataframe('data')
train_data = data_all[0]
test_data = data_all[1]
print("train_data")
print(train_data)
print("test_data")
print(test_data)

test_data_sheet = test_data_process(test_data)
test_data_sheet = test_data_sheet.permute(2, 1, 0)
train_data = train_data_resample(train_data)
train_data = train_data.permute(2, 1, 0)
test_dataset = test_Dataset_construct(test_data_sheet)
train_dataset = train_Dataset_construct(train_data)
with open('data/train_data.pkl', 'wb') as f:
    dill.dump(train_dataset, f)
with open('data/test_data.pkl', 'wb') as f:
    dill.dump(test_dataset, f)

end = time.time()
print(end - start)
