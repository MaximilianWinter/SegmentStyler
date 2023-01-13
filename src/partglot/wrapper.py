import numpy as np
import trimesh

from src.helper.paths import LOCAL_DATA_PATH, LOCAL_MODELS_PATH
from src.helper.visualization import (get_rnd_color,
                                      visualize_pointclouds_parts_partglot)
from src.partglot.utils.partglot_bspnet_preprocess import \
    convert_supersegs_to_pointclouds_simple
from src.partglot.utils.predict import (extract_pc2label,
                                        extract_reference_sample,
                                        get_attn_mask_objects,
                                        get_loaded_model, predict_ssegs2label,
                                        preprocess_point_cloud,
                                        segment_pc_with_labels, ssegs2input)
from src.partglot.utils.processing import print_stats


class PartSegmenter():
    """
    Class wrapper to handle full part segmentation pipeline
    """
    def __init__(self,
                 part_names=["back", "seat", "leg", "arm"],
                 sseg_count=25,
                 partglot_data_dir=LOCAL_DATA_PATH / "partglot",
                 partglot_model_path=LOCAL_MODELS_PATH / "partglot_pn_agnostic.ckpt"):
        self.partglot_data_dir, self.partglot_model_path = partglot_data_dir, partglot_model_path
        self.partglot, self.partglot_dm = self._load_partglot()
        self.label_cmap = [0xff0000, 0x00ff00, 0x0000ff, 0xff00ff, 0xffff00, 0x00ffff]
        self.sseg_cmap = [get_rnd_color() for i in range(1000)]
        self.sseg_count = sseg_count
        self.use_sseg_gt = False
        self.part_names = part_names
        print(f"PartSegmenter initialized with\n- sseg_count: {sseg_count}\n- partglot_data_dir: {partglot_data_dir}\n- partglot_model_path: {partglot_model_path}\n")
        
    def run_from_ref_data(self, sample_idx:int, use_sseg_gt:bool=True, cluster_tgt:str="vertices") -> tuple:
        """
        Returns a segmented point cloud based on the PartGlot CiC BSP-Net Dataset
            
            Parameters:
                sample_idx (int): sample index from CiC BSP-Net Dataset to be run
                use_sseg_gt (bool): defines if super segments should be used from 
                BSP-Net groud truth (True) or reclustered using cluster_tgt (False)
                !! NOTE: this ignores "cluster_tgt" if True !!
                cluster_tgt (str): clustering reference propertyfrom input mesh
                ("normals" | "vertices" | "vertices_and_normals") 
                
            Returns:
                partmap_idx_ranges (dict): part maps in our format (.jsonc)
                reordered_pc (np.array): point cloud reordered to map `partmap_idx_ranges` indices
                partmaps (np.array): point cloud regrouped in part-based labels
        """
        self.partmap_idx_ranges, self.reordered_pc, self.partmaps = \
            self._dummy_run(mesh=None, cluster_tgt=cluster_tgt, use_sseg_gt=use_sseg_gt, sample_idx=sample_idx)
        return self.partmap_idx_ranges, self.reordered_pc, self.partmaps
    
    def run_from_trimesh(self, mesh:trimesh.Trimesh, cluster_tgt:str="normals") -> tuple:
        """
        Returns a segmented point cloud based on custom input Trimesh mesh
            
            Parameters:
                mesh (trimesh.Trimesh): input mesh
                cluster_tgt (str): clustering reference property from input mesh
                ("normals" | "vertices" | "vertices_and_normals") 
                
            Returns:
                partmap_idx_ranges (dict): part maps in our format (.jsonc)
                reordered_pc (np.array): point cloud reordered to map `partmap_idx_ranges` indices
                partmaps (np.array): point cloud regrouped in part-based labels
        """
        self.partmap_idx_ranges, self.reordered_pc, self.partmaps = \
            self._dummy_run(mesh=mesh, cluster_tgt=cluster_tgt, use_sseg_gt=False, sample_idx=None)
        return self.partmap_idx_ranges, self.reordered_pc, self.partmaps
    
    def run_bspnet_data(self, bsp_idx:int) -> tuple:
        """
        Returns a segmented point cloud based on BSP-Net index (from BSP-Net input dataset)
            
            Parameters:
                bsp_idx (int): BSP-Net input dataset index
                
            Returns:
                partmap_idx_ranges (dict): part maps in our format (.jsonc)
                reordered_pc (np.array): point cloud reordered to map `partmap_idx_ranges` indices
                partmaps (np.array): point cloud regrouped in part-based labels
        """
        self.partmap_idx_ranges, self.reordered_pc, self.partmaps = \
            self._dummy_run(mesh=None, bsp_idx=bsp_idx, cluster_tgt=None, use_sseg_gt=False, sample_idx=None)
        return self.partmap_idx_ranges, self.reordered_pc, self.partmaps

    def visualize_labels(self, opacity=0.25, point_size=0.015):
        print(f'## About to visualize part labels')
        visualize_pointclouds_parts_partglot(self.partmaps, names=list(self.partmap_idx_ranges['mask_vertices'].keys()), part_colors=self.label_cmap, opacity=opacity, point_size=point_size)

    def visualize_ssegs(self, opacity=0.25, point_size=0.015):
        print(f'## About to visualize super segments - self.use_sseg_gt={self.use_sseg_gt}')
        visualize_pointclouds_parts_partglot(self.sup_segs, part_colors=self.sseg_cmap, opacity=opacity, point_size=point_size)

    def visualize_ssegs_bspnet(self, opacity=0.25, point_size=0.015):
        print(f'## About to visualize super segments (BSP-Net Groud Truth) - self.use_sseg_gt={self.use_sseg_gt}')
        visualize_pointclouds_parts_partglot(self.ref_sseg_data[0][0].cpu().numpy(), part_colors=self.sseg_cmap, opacity=opacity, point_size=point_size)

    def run_desinty_stats(self, use_sseg_gt=False):
        print_stats(self.reordered_pc, use_sseg_gt)
        
    def _dummy_run(self, mesh=None, cluster_tgt=None, use_sseg_gt=False, sample_idx=None, bsp_idx=None):
        print("Starting to run...\n")
        self.use_sseg_gt = use_sseg_gt
        self.ref_sseg_data, self.ref_mask_data = self._load_partglot_ref(sample_idx)
        self.mesh = mesh if mesh else self.ref_sseg_data
        self.sup_segs, self.pc2sup_segs = self._get_ssegs(self.mesh, cluster_tgt=cluster_tgt, bsp_idx=bsp_idx)
        self.sseg_count = self.pc2sup_segs.max() + 1 
        self._set_predict_input(self.use_sseg_gt)
        self.sup_segs2label = self._predict_ssegs2label()
        self.pc2label = self._get_pc2label()
        self.partmap_idx_ranges, self.reordered_pc = self._get_attn_map_objects() 
        self.partmaps = self._get_label_ssegs()
        self.run_desinty_stats(self.use_sseg_gt)
        print(f"Successfully ran part segmentation with use_sseg_gt={self.use_sseg_gt}\n")

        return self.partmap_idx_ranges, self.reordered_pc, self.partmaps

    
    def _load_partglot(self):
        return get_loaded_model(data_dir=self.partglot_data_dir, model_path=self.partglot_model_path)
    
    def _load_partglot_ref(self, sample_idx):
        if sample_idx is not None:
            return extract_reference_sample(self.partglot_dm.h5_data, sample_idx)
        else:
            return None, None
    
    def _get_ssegs(self, batch_point_cloud, normal_boundary="sphere", cluster_tgt="normals", bsp_idx=None):
        if bsp_idx is not None:
            sup_segs = convert_supersegs_to_pointclouds_simple(bsp_idx, normal_boundary=normal_boundary)
            pc2sup_segs = np.arange(sup_segs.shape[0])
            return sup_segs, pc2sup_segs
        else:
            return preprocess_point_cloud(batch_point_cloud, cluster_tgt=cluster_tgt)
        
    def _set_predict_input(self, use_sseg_gt=False):
        if use_sseg_gt:
            self.ssegs_batch = self.ref_sseg_data
            self.mask_batch = self.ref_mask_data 
            self.sup_segs = self.ref_sseg_data[0][0].cpu().numpy() # gt_super_segs / super_segs
        else:
            self.ssegs_batch, self.mask_batch = ssegs2input(self.sup_segs)
            
    def _predict_ssegs2label(self):
        return predict_ssegs2label(self.ssegs_batch, self.mask_batch, self.partglot_dm.word2int, self.partglot, self.part_names)
    
    def _get_pc2label(self):
        return extract_pc2label(self.sup_segs2label)
    
    def _get_attn_map_objects(self):
        return get_attn_mask_objects(self.ssegs_batch, self.pc2label) 
        
    def _get_label_ssegs(self):
        return segment_pc_with_labels(self.reordered_pc, self.partmap_idx_ranges)
        