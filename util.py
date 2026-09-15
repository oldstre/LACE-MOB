import os
import pickle
import random
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


class DataLoader:
    def __init__(self, x, y, batch_size, phys, social):
        self.batch_size = batch_size
        self.current_ind = 0
        self.size = len(x)
        padding = (batch_size - self.size % batch_size) % batch_size
        if padding:
            x = np.concatenate([x, np.repeat(x[-1:], padding, axis=0)])
            y = np.concatenate([y, np.repeat(y[-1:], padding, axis=0)])
            phys = np.concatenate([phys, np.repeat(phys[-1:], padding, axis=0)])
            social = np.concatenate([social, np.repeat(social[-1:], padding, axis=0)])
        self.x, self.y, self.phys, self.social = x, y, phys, social
        self.num_batch = len(x) // batch_size

    def shuffle(self):
        order = np.random.permutation(len(self.x))
        self.x, self.y = self.x[order], self.y[order]
        self.phys, self.social = self.phys[order], self.social[order]

    def get_iterator(self):
        self.current_ind = 0

        def iterator():
            while self.current_ind < self.num_batch:
                start = self.current_ind * self.batch_size
                end = (self.current_ind + 1) * self.batch_size
                yield self.x[start:end], self.y[start:end], self.phys[start:end], self.social[start:end]
                self.current_ind += 1

        return iterator()


class StandardScaler:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name):
    if str(name).startswith('cuda') and not torch.cuda.is_available():
        print('CUDA is unavailable; using CPU.')
        return torch.device('cpu')
    return torch.device(name)


def infer_dataset_metadata(dataset_dir):
    with np.load(os.path.join(dataset_dir, 'train.npz')) as data:
        x, y = data['x'], data['y']
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError('Expected four-dimensional x/y arrays.')
    return {
        'input_length': int(x.shape[1]), 'num_nodes': int(x.shape[2]),
        'in_dim': int(x.shape[3]), 'seq_length': int(y.shape[1]),
        'x_shape': tuple(x.shape), 'y_shape': tuple(y.shape),
    }


def resolve_adj_path(dataset_dir, path=None):
    if path:
        return path
    for name in ['adjacency_matrix.npy', 'adj_mx.pkl']:
        candidate = os.path.join(dataset_dir, name)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError('No adjacency matrix found in {}.'.format(dataset_dir))


def asym_adj(matrix):
    matrix = sp.coo_matrix(matrix)
    row_sum = np.asarray(matrix.sum(1)).flatten()
    inverse = np.zeros_like(row_sum, dtype=np.float32)
    nonzero = row_sum != 0
    inverse[nonzero] = np.power(row_sum[nonzero], -1)
    return sp.diags(inverse).dot(matrix).astype(np.float32).todense()


def load_adj(path, adjtype='doubletransition'):
    if path.endswith('.npy'):
        matrix = np.load(path)
        sensor_ids = [str(index) for index in range(matrix.shape[0])]
        sensor_map = {value: index for index, value in enumerate(sensor_ids)}
    elif path.endswith('.pkl'):
        with open(path, 'rb') as handle:
            try:
                loaded = pickle.load(handle)
            except UnicodeDecodeError:
                handle.seek(0)
                loaded = pickle.load(handle, encoding='latin1')
        if isinstance(loaded, tuple) and len(loaded) == 3:
            sensor_ids, sensor_map, matrix = loaded
        else:
            matrix = loaded
            sensor_ids = [str(index) for index in range(len(matrix))]
            sensor_map = {value: index for index, value in enumerate(sensor_ids)}
    else:
        raise ValueError('Unsupported adjacency format: {}'.format(path))
    matrix = np.asarray(matrix, dtype=np.float32)
    if adjtype == 'doubletransition':
        supports = [asym_adj(matrix), asym_adj(matrix.T)]
    elif adjtype == 'transition':
        supports = [asym_adj(matrix)]
    elif adjtype == 'identity':
        supports = [np.eye(matrix.shape[0], dtype=np.float32)]
    else:
        raise ValueError('Unsupported adjtype: {}'.format(adjtype))
    return sensor_ids, sensor_map, supports


def _clean(columns):
    return [column for column in columns if not str(column).lower().startswith('unnamed')]


def _key(value):
    return re.sub(r'[^a-z0-9]+', '', str(value).lower())


def _external_windows(metadata, sizes, phys_path, social_path, time_start, time_end,
                      drop_node_indices, exog_fill, node_align):
    if not phys_path or not social_path:
        raise ValueError('External conditioning requires --phys_path and --social_path.')
    physical, social = pd.read_csv(phys_path), pd.read_csv(social_path)
    physical.iloc[:, 0] = pd.to_datetime(physical.iloc[:, 0])
    social.iloc[:, 0] = pd.to_datetime(social.iloc[:, 0])
    phys_columns, social_columns = _clean(physical.columns[1:]), _clean(social.columns[1:])
    for index in sorted([int(v) for v in str(drop_node_indices).split(',') if v.strip()], reverse=True):
        phys_columns.pop(index)
        social_columns.pop(index)
    if node_align == 'name':
        if set(phys_columns) != set(social_columns):
            raise ValueError('External node columns do not match.')
        physical_values, social_values = physical[phys_columns], social[phys_columns]
    elif node_align == 'normalized_name':
        social_map = {_key(column): column for column in social_columns}
        keys = [_key(column) for column in phys_columns]
        if set(keys) != set(social_map):
            raise ValueError('Normalized external node columns do not match.')
        physical_values = physical[phys_columns]
        social_values = social[[social_map[key] for key in keys]]
    elif node_align == 'position' and len(phys_columns) == len(social_columns):
        physical_values, social_values = physical[phys_columns], social[social_columns]
    else:
        raise ValueError('Cannot align external node columns.')
    if len(phys_columns) != metadata['num_nodes']:
        raise ValueError('External node count does not match mobility data.')
    physical_values.index, social_values.index = physical.iloc[:, 0], social.iloc[:, 0]
    raw_length = sum(sizes.values()) + metadata['input_length'] + metadata['seq_length'] - 1
    timeline = pd.date_range(pd.Timestamp(time_start), pd.Timestamp(time_end), freq='h')
    if len(timeline) != raw_length:
        raise ValueError('Configured time range has {} hours; expected {}.'.format(len(timeline), raw_length))
    physical_values, social_values = physical_values.reindex(timeline), social_values.reindex(timeline)
    if exog_fill == 'zero':
        physical_values, social_values = physical_values.fillna(0), social_values.fillna(0)
    elif exog_fill == 'ffill_bfill':
        physical_values, social_values = physical_values.ffill().bfill(), social_values.ffill().bfill()
    elif physical_values.isna().any().any() or social_values.isna().any().any():
        raise ValueError('External signals contain missing timestamps.')
    phys = physical_values.to_numpy(np.float32)[..., None]
    social_array = social_values.to_numpy(np.float32)[..., None]
    total, length = sum(sizes.values()), metadata['input_length']
    phys = np.stack([phys[i:i + length] for i in range(total)])
    social_array = np.stack([social_array[i:i + length] for i in range(total)])
    train_end, val_end = sizes['train'], sizes['train'] + sizes['val']
    for values in [phys, social_array]:
        reference = np.sort(values[:train_end].reshape(-1))
        values[:] = np.searchsorted(reference, values, side='right') / float(len(reference))
    return phys, social_array, train_end, val_end


def load_dataset(dataset_dir, batch_size, valid_batch_size, test_batch_size,
                 phys_path=None, social_path=None, time_start=None, time_end=None,
                 drop_node_indices='', exog_fill='error', node_align='name', load_exogenous=True):
    data = {}
    for split in ['train', 'val', 'test']:
        with np.load(os.path.join(dataset_dir, split + '.npz')) as archive:
            data['x_' + split] = archive['x'].astype(np.float32)
            data['y_' + split] = archive['y'].astype(np.float32)
    metadata = infer_dataset_metadata(dataset_dir)
    sizes = {split: len(data['x_' + split]) for split in ['train', 'val', 'test']}
    if load_exogenous:
        phys, social, train_end, val_end = _external_windows(
            metadata, sizes, phys_path, social_path, time_start, time_end,
            drop_node_indices, exog_fill, node_align,
        )
        data.update({
            'phys_train': phys[:train_end], 'phys_val': phys[train_end:val_end], 'phys_test': phys[val_end:],
            'social_train': social[:train_end], 'social_val': social[train_end:val_end], 'social_test': social[val_end:],
            'phys_dim': 1,
        })
    else:
        data['phys_dim'] = 1
        for split, size in sizes.items():
            shape = (size, metadata['input_length'], metadata['num_nodes'], 1)
            data['phys_' + split] = np.zeros(shape, np.float32)
            data['social_' + split] = np.zeros(shape, np.float32)
    scaler = StandardScaler(data['x_train'][..., 0].mean(), data['x_train'][..., 0].std())
    for split in ['train', 'val', 'test']:
        data['x_' + split][..., 0] = scaler.transform(data['x_' + split][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, data['phys_train'], data['social_train'])
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size, data['phys_val'], data['social_val'])
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, data['phys_test'], data['social_test'])
    data['scaler'] = scaler
    return data


def _mask(labels, null_val):
    valid = (~torch.isnan(labels) if np.isnan(null_val) else labels != null_val).float()
    return torch.nan_to_num(valid / torch.clamp(valid.mean(), min=1e-8))


def masked_mae(prediction, labels, null_val=np.nan):
    return torch.mean(torch.nan_to_num(torch.abs(prediction - labels) * _mask(labels, null_val)))


def masked_mape(prediction, labels, null_val=np.nan):
    error = torch.abs((prediction - labels) / torch.clamp(torch.abs(labels), min=1e-5))
    return torch.mean(torch.nan_to_num(error * _mask(labels, null_val)))


def masked_rmse(prediction, labels, null_val=np.nan):
    return torch.sqrt(torch.mean(torch.nan_to_num(torch.square(prediction - labels) * _mask(labels, null_val))))


def metric(prediction, labels):
    return masked_mae(prediction, labels, 0.0).item(), masked_mape(prediction, labels, 0.0).item(), masked_rmse(prediction, labels, 0.0).item()
