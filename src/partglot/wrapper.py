from src.helper.visualization import visualize_pointclouds_parts_partglot
from src.helper.visualization import get_rnd_color
from src.partglot.utils.predict import get_loaded_model, extract_reference_sample, preprocess_point_cloud, ssegs2input, predict_ssegs2label, extract_pc2label, get_attn_mask_objects, segment_pc_with_labels
from src.partglot.utils.processing import print_stats
from src.helper.paths import LOCAL_MODELS_PATH, LOCAL_DATA_PATH


class PartSegmenter():
    """
    Class wrapper to handle full part segmentation pipeline
    """
    def __init__(self,
                 sseg_count=25,
                 partglot_data_dir=LOCAL_DATA_PATH / "partglot",
                 partglot_model_path=LOCAL_MODELS_PATH / "partglot_pn_agnostic.ckpt"):
        self.partglot_data_dir, self.partglot_model_path = partglot_data_dir, partglot_model_path
        self.partglot, self.partglot_dm = self._load_partglot()
        self.label_cmap = [0xff0000, 0x00ff00, 0x0000ff, 0xff00ff, 0xffff00, 0x00ffff]
        self.sseg_cmap = [get_rnd_color() for i in range(1000)]
        self.sseg_count = sseg_count
        self.use_sseg_gt = False
        print(f"PartSegmenter initialized with\n- sseg_count: {sseg_count}\n- partglot_data_dir: {partglot_data_dir}\n- partglot_model_path: {partglot_model_path}\n")
        
    def run_from_ref_data(self, sample_idx:int, use_sseg_gt=True, cluster_tgt="vertices") -> tuple:
        """
        Returns a segmented point cloud based on the PartGlot CiC BSP-Net Dataset
            
            Parameters:
                sample_idx (int): sample index from CiC BSP-Net Dataset to be run
                use_sseg_gt (bool): defines if super segments should be used from 
                BSP-Net groud truth or reclustered (currently only with K-Means)
                
            Returns:
                final_mask (dict): attn maps in our format
                final_pc (np.array): point cloud reordered to map final_mask indices
                label_ssegs (np.array): point cloud regrouped in part based labels
        """
        self.use_sseg_gt = use_sseg_gt
        self.ref_sseg_data, self.ref_mask_data = self._load_partglot_ref(sample_idx)
        self.sup_segs, self.pc2sup_segs = self._get_ssegs(self.ref_sseg_data, cluster_tgt=cluster_tgt)
        self._set_predict_input(self.use_sseg_gt)
        self.sup_segs2label = self._predict_ssegs2label()
        self.pc2label = self._get_pc2label()
        self.final_mask, self.final_pc = self._get_attn_map_objects() 
        self.label_ssegs = self._get_label_ssegs()
        self.run_desinty_stats(self.use_sseg_gt)
        print(f"Successfully ran part segmentation with use_sseg_gt={self.use_sseg_gt}\n")
        return self.final_mask, self.final_pc, self.label_ssegs
    
    def run_from_trimesh(self, mesh, cluster_tgt="normals") -> tuple:
        """
        Returns a segmented point cloud based on the PartGlot CiC BSP-Net Dataset
            
            Parameters:
                sample_idx (int): sample index from CiC BSP-Net Dataset to be run
                use_sseg_gt (bool): defines if super segments should be used from 
                BSP-Net groud truth or reclustered (currently only with K-Means)
                
            Returns:
                final_mask (dict): attn maps in our format
                final_pc (np.array): point cloud reordered to map final_mask indices
                label_ssegs (np.array): point cloud regrouped in part based labels
        """
        # self.ref_sseg_data, self.ref_mask_data = self._load_partglot_ref(sample_idx)
        self.mesh = mesh
        self.sup_segs, self.pc2sup_segs = self._get_ssegs(self.mesh, cluster_tgt=cluster_tgt)
        self._set_predict_input()
        self.sup_segs2label = self._predict_ssegs2label()
        self.pc2label = self._get_pc2label()
        self.final_mask, self.final_pc = self._get_attn_map_objects() 
        self.label_ssegs = self._get_label_ssegs()
        self.run_desinty_stats()
        print(f"Successfully ran part segmentation\n")
        return self.final_mask, self.final_pc, self.label_ssegs

    def visualize_labels(self, opacity=0.25, point_size=0.015):
        print(f'## About to visualize part labels')
        visualize_pointclouds_parts_partglot(self.label_ssegs, names=list(self.final_mask['mask_vertices'].keys()), part_colors=self.label_cmap, opacity=opacity, point_size=point_size)

    def visualize_ssegs(self, opacity=0.25, point_size=0.015):
        print(f'## About to visualize super segments - self.use_sseg_gt={self.use_sseg_gt}')
        visualize_pointclouds_parts_partglot(self.sup_segs, part_colors=self.sseg_cmap, opacity=opacity, point_size=point_size)

    def visualize_ssegs_bspnet(self, opacity=0.25, point_size=0.015):
        print(f'## About to visualize super segments (BSP-Net Groud Truth) - self.use_sseg_gt={self.use_sseg_gt}')
        visualize_pointclouds_parts_partglot(self.ref_sseg_data[0][0].cpu().numpy(), part_colors=self.sseg_cmap, opacity=opacity, point_size=point_size)

    def run_desinty_stats(self, use_sseg_gt=False):
        print_stats(self.final_pc, use_sseg_gt)
        
    def _load_partglot(self):
        return get_loaded_model(data_dir=self.partglot_data_dir, model_path=self.partglot_model_path)
    
    def _load_partglot_ref(self, sample_idx):
        return extract_reference_sample(self.partglot_dm.h5_data, sample_idx)
    
    def _get_ssegs(self, batch_point_cloud, cluster_tgt="normals"):
        return preprocess_point_cloud(batch_point_cloud, cluster_tgt=cluster_tgt)
        
    def _set_predict_input(self, use_sseg_gt=False):
        if use_sseg_gt:
            self.ssegs_batch = self.ref_sseg_data
            self.mask_batch = self.ref_mask_data 
            self.sup_segs = self.ref_sseg_data[0][0].cpu().numpy() # gt_super_segs / super_segs
        else:
            self.ssegs_batch, self.mask_batch = ssegs2input(self.sup_segs)
            
    def _predict_ssegs2label(self):
        return predict_ssegs2label(self.ssegs_batch, self.mask_batch, self.partglot_dm.word2int, self.partglot)
    
    def _get_pc2label(self):
        return extract_pc2label(self.sup_segs2label)
    
    def _get_attn_map_objects(self):
        return get_attn_mask_objects(self.ssegs_batch, self.pc2label) 
        
    def _get_label_ssegs(self):
        return segment_pc_with_labels(self.final_pc, self.final_mask)
        