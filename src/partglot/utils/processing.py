import torch
import numpy as np


def sort_arrays(arrays):
    ref_array = arrays[0]
    sorted_indices = ref_array.argsort()
    out = []
    for a in arrays:
        out.append(a[sorted_indices])
    return out

def random_sample_array(arr: np.array, size: int = 1, with_replacement:bool=True) -> np.array:
    if with_replacement:
        while len(arr) < size:
            arr = np.concatenate([arr, arr])
    return arr[np.random.choice(len(arr), size=size, replace=False)]

def cluster_supsegs(sorted_labels, sorted_pc, sup_seg_size=512):
    sup_segs, labels = [], []
    for lbl in np.unique(sorted_labels):
        indices = np.where(sorted_labels==lbl)[0]
        tmp_pc = sorted_pc[indices]
        sup_segs.append(random_sample_array(tmp_pc, sup_seg_size))
        labels.append(lbl)
    return np.array(sup_segs), np.array(labels)

def vstack2dim(data:np.array, dim=2):
    if len(data.shape) <= dim:
        return data
    else:
        data = np.vstack(convert2np(data))
        return vstack2dim(data=data, dim=dim)


def print_stats(pc_final:torch.tensor, use_bsp_ssegs_gt:bool):
    
    pc_final = vstack2dim(convert2np(pc_final))
    
    mask = (pc_final != np.array([0,0,0])).max(axis=1)
    non_zero_pc = pc_final[mask]

    unique_point_perc = np.unique(pc_final, axis=0).shape[0] / vstack2dim(pc_final).shape[0]
    unique_point_perc_non_zero = np.unique(non_zero_pc, axis=0).shape[0] / vstack2dim(non_zero_pc).shape[0]
    print(f"Running point cloud stats with use_bsp_ssegs_gt={use_bsp_ssegs_gt}")
    print(f"- unique point percentage: {unique_point_perc:.1%}")
    print(f"- unique point percentage (non-zeros): {unique_point_perc_non_zero:.1%}\n")


def convert2np(tensor):
    if torch.is_tensor(tensor):
        tensor = tensor.cpu().numpy()
    return tensor