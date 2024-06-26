import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime

def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr
    
def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

# def centerize_vary_length_series(x):
#     prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
#     suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
#     offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
#     rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
#     offset[offset < 0] += x.shape[1]
#     column_indices = column_indices - offset[:, np.newaxis]
#     return x[rows, column_indices]

def centerize_vary_length_series(x, mask=None):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    if mask != None:
        return x[rows, column_indices], mask[rows, column_indices]
    else:
        return x[rows, column_indices]

def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res

def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]

def generate_mask(data, p = 0.5, remain = 0):
    B, T, C = data.shape
    mask = np.empty_like(data)
    ### this function basically creates a mask for the non zero or non missing values in each channel based on the value of p. In the calling function, they have set p=1 so it will set the mask with values 1 and the missing values are set to nan
    # this is taking ages because it is peforming this for loop across all data points!!!
    for b in range(B):
        ts = data[b, :, 0]
        et_num = ts[~np.isnan(ts)].size - remain
        total, num = et_num * C, round(et_num * C * p)

        while True:
            i_mask = np.zeros(total)
            i_mask[random.sample(range(total), num)] = 1
            i_mask = i_mask.reshape(et_num, C)
            if 1 not in i_mask.sum(axis = 0) and 0 not in i_mask.sum(axis = 0):
                break
            break

        i_mask = np.concatenate((i_mask, np.ones((remain, C))), axis = 0)
        mask[b, ~np.isnan(ts), :] = i_mask
        mask[b, np.isnan(ts), :] = np.nan

    # mask = np.concatenate([random.sample(range(total), num) for _ in range(B)])
    # matrix = np.zeros((B, total))
    # matrix[(np.arange(B).repeat(num), mask)] = 1.0
    # matrix = matrix.reshape(B, T, C)
    # return matrix
    return mask

def normalize_with_mask(train, mask_tr, test, mask_te, scaler):
    ## set the masked values (0) to nan in both train and test; perform the usual scaler ; and then set the originally set missing values to 0 in the standardized data
    train[mask_tr == 0], test[mask_te == 0] = np.nan, np.nan
    scaler = scaler.fit(train.reshape(-1, train.shape[-1]))
    train = scaler.transform(train.reshape(-1, train.shape[-1])).reshape(train.shape)
    test = scaler.transform(test.reshape(-1, test.shape[-1])).reshape(test.shape)
    train[mask_tr == 0], test[mask_te == 0] = 0, 0
    return train, test