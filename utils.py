import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from time import perf_counter
from _calculate import rbf_sim


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_citation(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load Citation Networks Datasets.
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str.lower(), names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    adj, features = preprocess_citation(adj, features, normalization)

    # porting to pytorch
    features = torch.FloatTensor(np.array(features.todense())).float()
    labels = torch.LongTensor(labels)
    labels = torch.max(labels, dim=1)[1]
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test

def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)

    precompute_time = perf_counter()-t
    return features, precompute_time

def build_adj_rbf(features, adj, alpha=1.8):
    # slow because of python loop
    n_nodes, feat_dim = features.shape

    indices = adj.coalesce().indices()
    values = adj.coalesce().values()
    shape = (n_nodes, n_nodes)

    n_edges = indices.shape[1]

    for i in range(n_edges):
        x = indices[0][i]
        y = indices[1][i]
        values[i] = values[i] * torch.exp(-(torch.norm((features[x] - features[y]), p=2)/(2*alpha*alpha)))
    
    return torch.sparse.FloatTensor(indices, values, shape)


def build_adj_rbf_fast(features, adj, alpha=1.8):
    # with c extension built by swig
    # much faster than python loop
    n_nodes, feat_dim = features.shape

    indices = adj.coalesce().indices()
    numpy_indices = indices.cpu().numpy()
    values = adj.coalesce().values()
    shape = (n_nodes, n_nodes)

    n_edges = indices.shape[1]
    numpy_values = values.cpu().numpy()

    rbf_sim(numpy_indices[0], numpy_indices[1], numpy_values, features.cpu().numpy(), alpha)
    values = torch.from_numpy(numpy_values).cuda()

    return torch.sparse.FloatTensor(indices, values, shape)


def build_adj_cos(features, adj):
    # n_nodes, _ = features.shape

    # indices = adj.coalesce().indices()
    # values = adj.coalesce().values()
    # shape = (n_nodes, n_nodes)

    # n_edges = indices.shape[1]

    # for i in range(n_edges):
    #     x = indices[0][i]
    #     y = indices[1][i]
    #     values[i] = values[i] * (torch.dot(features[x], features[y]) / (torch.norm(features[x], p=2) * torch.norm(features[y], p=2)).add(1e-10))

    # return torch.sparse.FloatTensor(indices, values, shape)
    pass

def build_adj_lle(features, adj):
    pass


def Lp_dis(X, Y, p=2):
    # not normalized into [0,1]
    pass


def cos_dis(X, Y):
    norm_X = torch.norm(X, p=2, dim=1).add(1e-10)
    norm_Y = torch.norm(Y, p=2, dim=1).add(1e-10)
    cosine = torch.div(torch.div((X*Y).sum(1), norm_X), norm_Y)
    dist = 1 - cosine.unsqueeze(-1)
    return dist


def sgc_precompute_npa(features, adj, hops, T, alpha, fast):
    t = perf_counter()
    Adj = adj
    pre_features = features

    for i in range(hops):
        if fast:
            Adj = build_adj_rbf_fast(pre_features, adj, alpha)
        else:
            Adj = build_adj_rbf(pre_features, adj, alpha)
        features = torch.spmm(Adj, pre_features)
        w = cos_dis(features, pre_features)
        w = torch.exp(w/T) / (torch.exp(w/T) + torch.exp((1-w)/T))
        features = w * features + (1 - w) * pre_features
        pre_features = features

    precompute_time = perf_counter()-t
    return features, precompute_time


def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)
