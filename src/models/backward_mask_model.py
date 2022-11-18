import kaolin as kal

from src.utils.Normalization import MeshNormalizer
from src.models import Text2MeshOriginal
from src.models.mask_backward import MaskBackward


class Text2MeshBackwardMask(Text2MeshOriginal):

    def __init__(self, args, base_mesh):
        super().__init__(args, base_mesh)
        print("Initializing Text2MeshBackwardMask...")
        self.mask_backward = MaskBackward.apply

    def forward(self, vertices):
        # Prop. through MLP
        pred_rgb, pred_normal = self.mlp(vertices)

        # Do backward masking
        mask_rgb = 1 - self.load_mask(pred_rgb) # inverting the mask, because the mask is originally defined for the penalizing loss
        pred_rgb = self.mask_backward(pred_rgb, mask_rgb)

        mask_disp = 1 - self.load_mask(pred_normal)
        pred_normal = self.mask_backward(pred_normal, mask_disp)
            
        # Get stylized mesh
        self.base_mesh.face_attributes = self.prior_color + kal.ops.mesh.index_vertices_by_faces(
            pred_rgb.unsqueeze(0),
            self.base_mesh.faces)

        if self.args.optimize_displacement:
            self.base_mesh.vertices = self.base_mesh_vertices + self.base_mesh.vertex_normals * pred_normal
        else:
            self.base_mesh.vertices = self.base_mesh_vertices

        MeshNormalizer(self.base_mesh)()

        # Rendering
        rendered_images, elev, azim = self.renderer.render_front_views(self.base_mesh, num_views=self.args.n_views,
                                                                show=self.args.show,
                                                                center_azim=self.args.frontview_center[0],
                                                                center_elev=self.args.frontview_center[1],
                                                                std=self.args.frontview_std,
                                                                return_views=True,
                                                                background=self.background)
        geo_renders = None
        if self.args.geoloss:
            self.base_mesh.face_attributes = kal.ops.mesh.index_vertices_by_faces(self.default_color.unsqueeze(0),
                                                                                    self.base_mesh.faces)
            
            geo_renders, elev, azim = self.renderer.render_front_views(self.base_mesh, num_views=self.args.n_views,
                                                                    show=self.args.show,
                                                                    center_azim=self.args.frontview_center[0],
                                                                    center_elev=self.args.frontview_center[1],
                                                                    std=self.args.frontview_std,
                                                                    return_views=True,
                                                                    background=self.background)

        # Augmentations and CLIP encoding
        encoded_renders_dict = self.clip_with_augs.get_encoded_renders(rendered_images, geo_renders)

        color_reg = self.get_color_reg(pred_rgb)

        return {"encoded_renders": encoded_renders_dict, "rendered_images": rendered_images, "color_reg": color_reg}
