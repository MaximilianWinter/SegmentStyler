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

def get_attn_mask_objects(pc, pc2label, part_names):
    """
    Returns ordered point cloud and mask indices in our format.
    """
    stacked_pc = np.vstack(np.vstack(pc))
    
    arg_sort = pc2label.argsort()
    
    out_pc2label, out_pg_pc = pc2label[arg_sort], np.vstack(stacked_pc)[arg_sort]

    mask = {}
    for i, pn in enumerate(part_names):
        tmp = np.where(out_pc2label == i)[0]
        print(tmp.shape)
        if tmp.shape[0] == 0:
            continue
        mask[pn] = [tmp.min(), tmp.max()]
    
    return {"mask_vertices": mask}, out_pg_pc

def vstack2dim(data, dim=2):
    if len(data.shape) <= dim:
        return data
    else:
        data = np.vstack(data)
        return vstack2dim(data=data, dim=dim)

