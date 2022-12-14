from src.helper.visualization import visualize_pointclouds_parts_partglot
from src.helper.visualization import get_rnd_color
from src.partglot.utils.neural_utils import tokenizing
from src.partglot.utils.predict import get_loaded_model
from src.partglot.utils.processing import cluster_supsegs, get_attn_mask_objects, vstack2dim

import numpy as np
import torch
from sklearn.cluster import KMeans


if __name__ == '__main__':
        
    part_names = ["back", "seat", "leg", "arm"]
    sseg_cmap = [get_rnd_color() for i in range(1000)]

    part_semantic_groups = {
        "back": ["back"],
        "seat": ["seat"],
        "leg": ["leg", "wheel", "base"],
        "arm": ["arm"],
    }

    sample_idx = 1
    use_bsp_ssegs_gt = False
    n_ssseg_custom = 25
    opacity = 0.25

    np.random.seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    label_cmap = [0xff0000, 0x00ff00, 0x0000ff, 0xff00ff, 0xffff00, 0x00ffff]


    model_dir = "/home/bellatini/DL3D-Practical/models/pn_agnostic.ckpt"
    # data_dir = "/home/bellatini/DL3D-Practical/data/partglot"
    data_dir = "/home/bellatini/DL3D-Practical/data/partglot_100"

    # LOAD MODEL AND REFERENCE DATASET
    partglot, partglot_dm = get_loaded_model(data_dir=data_dir, model_path=model_dir)
    batch_data = torch.from_numpy(partglot_dm.h5_data['data'][sample_idx:sample_idx+1]).unsqueeze(dim=1).float().to(device)
    mask_data = torch.from_numpy(partglot_dm.h5_data['mask'][sample_idx:sample_idx+1]).unsqueeze(dim=1).float().to(device)

    # sup_segs2label, pc2label = partglot.get_attn_maps()[sample_idx]
    # segs, masks = partglot_dm.h5_data['data'][sample_idx], partglot_dm.h5_data['mask'][sample_idx]

    partglot.to(device)

    # CLUSTER POINT CLOUD INTO SSEGS (KMEANS)
    pc = vstack2dim(batch_data.cpu().numpy())
    kmeans = KMeans(n_clusters=n_ssseg_custom, random_state=1).fit(pc)
    pc2sup_segs_kmeans = kmeans.labels_
    # sorted_labels, sorted_pc = sort_arrays((pc2sup_segs_kmeans, pc))
    # sorted_ssegs, pc2sup_segs = cluster_supsegs(sorted_labels, sorted_pc)
    sup_segs, pc2sup_segs = cluster_supsegs(pc2sup_segs_kmeans, pc)

    # SET VARIABLES FOR PREDICTION (REF POINT CLOUDS OR CUSTOM ONES)
    if use_bsp_ssegs_gt:
        final_ssegs_batch = batch_data
        final_mask_batch = mask_data 
        sup_segs = batch_data[0][0].cpu().numpy() # gt_super_segs / super_segs
    else:
        final_ssegs_batch = torch.from_numpy(np.array([[sup_segs]])).float().to(device) 
        final_mask_batch = torch.from_numpy(np.array([[np.ones(final_ssegs_batch.shape[2])]])).float().to(device) 
        
    # GET ATTN MAPS PER SSEG (sseg2label)
    attn_maps = []
    for pn in part_names:
        text_embeddings = tokenizing(partglot_dm.word2int, f"chair with a {pn}").to(device)[None].expand(
            1, -1
        )
        tmp = partglot.forward(
            final_ssegs_batch, # custom_ssegs_batch / batch_data
            final_mask_batch, # custom_mask_batch / mask_data
            text_embeddings, True)
        attn_maps.append(tmp)
        
    attn_maps_concat = torch.cat(attn_maps).max(0)[1].cpu().numpy()

    sup_segs2label = np.squeeze(attn_maps_concat)
    sup_segs2label

    # EXPAND ATTN MAPS TO POINT-LEVEL GRANULARITY (pc2label)
    pc2label=[] # pc2sup_segs: is actually pc2label
    for lbl in sup_segs2label:
        tmp = np.ones(512) * lbl
        pc2label.append(tmp)
        
    pc2label = np.concatenate(pc2label).astype(int)

    assign_ft = lambda x: sup_segs2label[x]

    # pc2label_sorted = assign_ft(pc2sup_segs.astype(int))
    # import numpy as np
    # pc2label_prefinal = []
    # for lbl in sup_segs2label:
    #     tmp = np.ones(512)*lbl
    #     pc2label_prefinal.append(tmp)
    # pc2label_prefinal = np.concatenate(pc2label_prefinal)

    # VISUALIZE PART SEGMENTATION LABELS
    pc_final = vstack2dim(final_ssegs_batch.cpu().numpy())
    # pc_final = vstack2dim(vstack2dim(sorted_ssegs))

    # final_mask, final_pc = get_attn_mask_objects(super_segs, pc2label)
    final_mask, final_pc = get_attn_mask_objects(pc_final, pc2label, part_names) # pc2sup_segs: is actually pc2label

    mask = (pc_final != np.array([0,0,0])).max(axis=1)
    non_zero_pc = pc_final[mask]

    unique_point_perc = np.unique(pc_final, axis=0).shape[0] / vstack2dim(pc_final).shape[0]
    unique_point_perc_non_zero = np.unique(non_zero_pc, axis=0).shape[0] / vstack2dim(non_zero_pc).shape[0]
    print("use_bsp_ssegs_gt:", use_bsp_ssegs_gt)
    print(f"unique point percentage: {unique_point_perc:.1%}")
    print(f"unique point percentage (non-zeros): {unique_point_perc_non_zero:.1%}")

    out = []
    for s,f in final_mask['mask_vertices'].values():
        tmp = final_pc[s:f].astype(float)
        out.append(tmp)
        
    visualize_pointclouds_parts_partglot(np.array(out), names=list(final_mask['mask_vertices'].keys()), part_colors=label_cmap, opacity=opacity)
