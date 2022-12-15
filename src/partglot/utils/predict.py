import torch
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

from src.utils.processing import zip_arrays
from src.partglot.datamodules.partglot_datamodule import PartglotDataModule
from src.partglot.models.pn_agnostic import PNAgnostic
from src.utils.utils import device
from src.partglot.utils.processing import cluster_supsegs, vstack2dim, convert2np
from src.partglot.utils.neural_utils import tokenizing


def get_loaded_model(data_dir, model_path="models/partglot_pn_agnostic.ckpt", batch_size=1):
    datamodule = PartglotDataModule(batch_size=batch_size,
        only_correct=True,
        only_easy_context=False,
        max_seq_len=33,
        only_one_part_name=True,
        seed = 12345678,
        split_sizes = [0.8, 0.1, 0.1],
        balance=True,
        data_dir=data_dir)

    model = PNAgnostic(text_dim=64,
            embedding_dim=100,
            sup_segs_dim=64,
            lr=1e-3,
            data_dir=data_dir,
            word2int=datamodule.word2int,
            total_steps=1,
            measure_iou_every_epoch=True,
            save_pred_label_every_epoch=False)

    ckpt = torch.load(model_path)
    if "state_dict" in ckpt:
        # print("write state dict")
        ckpt = ckpt["state_dict"]

    model.load_state_dict(ckpt)
    
    return model.to(device), datamodule

def extract_reference_sample(h5_data, sample_idx=0):
    batch_data = torch.from_numpy(h5_data['data'][sample_idx:sample_idx+1]).unsqueeze(dim=1).float().to(device)
    mask_data = torch.from_numpy(h5_data['mask'][sample_idx:sample_idx+1]).unsqueeze(dim=1).float().to(device)
    return batch_data, mask_data

def preprocess_point_cloud(mesh, cluster_method="kmeans", cluster_tgt="normals", n_ssseg_custom=25, random_state=0):
    
    if cluster_tgt == "normals":
        cluster_tgt, coordinates = mesh.vertex_normals, mesh.vertices
    elif cluster_tgt == "vertices":
        cluster_tgt, coordinates = vstack2dim(mesh), vstack2dim(mesh)
    elif cluster_tgt == "vertices_and_normals":
        cluster_tgt, coordinates = zip_arrays(mesh.vertex_normals, mesh.vertices), mesh.vertices
    else:
        raise NotImplementedError
    
    if cluster_method == "dbscan":
        clusterizer  = DBSCAN(eps=1.25)
    elif cluster_method == "kmeans":
        clusterizer = KMeans(n_clusters=n_ssseg_custom, random_state=0)
    else:
        raise NotImplementedError
    
    pc2sup_segs_kmeans = clusterizer.fit(cluster_tgt).labels_
    sup_segs, pc2sup_segs = cluster_supsegs(pc2sup_segs_kmeans, coordinates)
    
    return sup_segs, pc2sup_segs

def ssegs2input(sup_segs):
    data_batch = torch.from_numpy(np.array([[sup_segs]])).float().to(device) 
    mask_batch = torch.from_numpy(np.array([[np.ones(data_batch.shape[2])]])).float().to(device) 
    return data_batch, mask_batch

def predict_ssegs2label(ssegs_batch, mask_batch, word2int, partglot, part_names=["back", "seat", "leg", "arm"]):
    attn_maps = []
    for pn in part_names:
        text_embeddings = tokenizing(word2int, f"chair with a {pn}").to(device)[None].expand(
            1, -1
        )
        tmp = partglot.forward(
            ssegs_batch, # custom_ssegs_batch / batch_data
            mask_batch, # custom_mask_batch / mask_data
            text_embeddings, True)
        attn_maps.append(tmp)
        
    attn_maps_concat = torch.cat(attn_maps).max(0)[1].cpu().numpy()

    sup_segs2label = np.squeeze(attn_maps_concat)
    return sup_segs2label


def extract_pc2label(ssegs2label):
    pc2label=[] 
    for lbl in ssegs2label:
        tmp = np.ones(512) * lbl
        pc2label.append(tmp)
        
    pc2label = np.concatenate(pc2label).astype(int)
    
    return pc2label


def get_attn_mask_objects(pc:np.array, pc2label:np.array, part_names=["back", "seat", "leg", "arm"]):
    """
    Returns ordered point cloud and mask indices in our format.
    """
    stacked_pc = vstack2dim(pc)
    # pc_final = vstack2dim(final_ssegs_batch.cpu().numpy())
    
    
    arg_sort = pc2label.argsort()
    
    out_pc2label, out_pg_pc = pc2label[arg_sort], np.vstack(stacked_pc)[arg_sort]

    mask = {}
    for i, pn in enumerate(part_names):
        tmp = np.where(out_pc2label == i)[0]
        if tmp.shape[0] == 0:
            continue
        mask[pn] = [tmp.min(), tmp.max()]
    
    return {"mask_vertices": mask}, out_pg_pc


def segment_pc_with_labels(final_pc, final_mask):
    label_ssegs = []
    for s,f in final_mask['mask_vertices'].values():
        tmp = final_pc[s:f].astype(float)
        label_ssegs.append(tmp)
    return np.array(label_ssegs, dtype=object)