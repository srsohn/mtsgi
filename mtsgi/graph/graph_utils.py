import numpy as np


def _sample_int_layer_wise(nbatch, high, low):
    assert(high.ndim==1 and low.ndim==1)
    ndim = len(high)
    out_list = []
    for d in range(ndim):
        out_list.append( np.random.randint(low[d], high[d]+1, (nbatch,1 ) ) )
    return np.concatenate(out_list, axis=1)


def _transform(input_arr, mapping):
    if input_arr.ndim==1:
        input_arr = np.expand_dims(input_arr, -1)
    return np.take_along_axis(mapping, input_arr, axis=1)


def _to_multi_hot(index_tensor, max_dim):
    # number-to-onehot or numbers-to-multihot
    if len(index_tensor.shape)==1:
        out = (np.expand_dims(index_tensor, axis=1) == \
               np.arange(max_dim).reshape(1, max_dim))
    else:
        out = (index_tensor == np.arange(max_dim).reshape(1, max_dim))
    return out


from dataclasses import dataclass

@dataclass
class SubtaskGraph:
    numP: np.ndarray
    numA: np.ndarray
    ind_to_id: np.ndarray
    id_to_ind: np.ndarray
    rmag: np.ndarray
    W_a: np.ndarray
    ORmat: np.ndarray
    tind_by_layer: list
    tind_list: list

    def __init__(self):
        self.numP = self.numA = None
        self.ind_to_id = self.id_to_ind = None
        self.rmag = self.W_a = self.ORmat = None
        self.tind_by_layer = self.tind_list = None


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def batch_bin_encode(bin_tensor):
    dim = len(bin_tensor.shape)
    feat_dim = bin_tensor.shape[-1]
    bias = 0
    unit = 50
    if dim == 2:
        NB = bin_tensor.shape[0]
        output = [0] * NB
        num_iter = feat_dim//unit + 1
        for i in range(num_iter):
            ed = min(feat_dim, bias + unit)
            out = batch_bin_encode_64(bin_tensor[:, bias:ed])
            out_list = out.tolist()
            output = [output[j] * pow(2, unit) + val for j, val in enumerate(out_list)]
            bias += unit
            if ed==feat_dim:
                break
        return output

    elif dim == 1:
        output = 0
        num_iter = feat_dim//unit + 1
        for i in range(num_iter):
            ed = min(feat_dim, bias + unit)
            out = batch_bin_encode_64(bin_tensor[bias:ed])
            output = output * pow(2, unit) + out
            bias += unit
            if ed==feat_dim:
                break
        return output

    else:
        raise ValueError("dim = %s" % dim)


def batch_bin_encode_64(bin_tensor):
    # bin_tensor: Nbatch x dim
    assert isinstance(bin_tensor, np.ndarray)
    return bin_tensor.dot(
        (1 << np.arange(bin_tensor.shape[-1])).astype(np.int64)
    )
