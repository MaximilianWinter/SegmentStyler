from src.helper.visualization import visualize_pointclouds_parts_partglot
from src.helper.visualization import get_rnd_color
from src.partglot.utils.predict import get_loaded_model, extract_reference_sample, preprocess_point_cloud, ssegs2input, predict_ssegs2label, extract_pc2label, get_attn_mask_objects, segment_pc_with_labels
from src.utils.config import get_parser


if __name__ == '__main__':

    args = get_parser().parse_args()
    sseg_cmap = [get_rnd_color() for i in range(1000)]
    sample_idx = 1
    use_bsp_ssegs_gt = False
    n_ssseg_custom = 25
    opacity = 0.25
    label_cmap = [0xff0000, 0x00ff00, 0x0000ff, 0xff00ff, 0xffff00, 0x00ffff]

    # LOAD MODEL AND REFERENCE DATASET
    partglot, partglot_dm = get_loaded_model(data_dir=args.partglot_data_dir, model_path=args.partglot_model_path)
    batch_data, mask_data = extract_reference_sample(partglot_dm.h5_data, sample_idx)

    # CLUSTER POINT CLOUD INTO SSEGS (KMEANS)
    sup_segs, pc2sup_segs = preprocess_point_cloud(batch_data)
    
    # SET VARIABLES FOR PREDICTION (REF POINT CLOUDS OR CUSTOM ONES)
    if use_bsp_ssegs_gt:
        final_ssegs_batch = batch_data
        final_mask_batch = mask_data 
        sup_segs = batch_data[0][0].cpu().numpy() # gt_super_segs / super_segs
    else:
        final_ssegs_batch, final_mask_batch = ssegs2input(sup_segs)
        
    # GET ATTN MAPS PER SSEG (sseg2label)
    sup_segs2label = predict_ssegs2label(final_ssegs_batch, final_mask_batch, partglot_dm.word2int, partglot)

    # EXPAND ATTN MAPS TO POINT-LEVEL GRANULARITY (pc2label)
    pc2label = extract_pc2label(sup_segs2label)

    # VISUALIZE PART SEGMENTATION LABELS
    final_mask, final_pc = get_attn_mask_objects(final_ssegs_batch, pc2label) 

    label_ssegs = segment_pc_with_labels(final_pc, final_mask)

    visualize_pointclouds_parts_partglot(label_ssegs, names=list(final_mask['mask_vertices'].keys()), part_colors=label_cmap, opacity=opacity)
