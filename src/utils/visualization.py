import numpy as np
from pathlib import Path
from src.utils.utils import getRotMat
from src.utils.mesh import trimMesh


# ================== VISUALIZATION =======================
# Back out camera parameters from view transform matrix
def extract_from_gl_viewmat(gl_mat):
    gl_mat = gl_mat.reshape(4, 4)
    s = gl_mat[0, :3]
    u = gl_mat[1, :3]
    f = -1 * gl_mat[2, :3]
    coord = gl_mat[:3, 3]  # first 3 entries of the last column
    camera_location = np.array([-s, -u, f]).T @ coord
    target = camera_location + f * 10  # any scale
    return camera_location, target


def psScreenshot(vertices, faces, axis, angles, save_path, name="mesh", frame_folder="frames", scalars=None,
                 colors=None,
                 defined_on="faces", highlight_faces=None, highlight_color=[1, 0, 0], highlight_radius=None,
                 cmap=None, sminmax=None, cpos=None, clook=None, save_video=False, save_base=False,
                 ground_plane="tile_reflection", debug=False, edge_color=[0, 0, 0], edge_width=1, material=None):
    import polyscope as ps

    ps.init()
    # Set camera to look at same fixed position in centroid of original mesh
    # center = np.mean(vertices, axis = 0)
    # pos = center + np.array([0, 0, 3])
    # ps.look_at(pos, center)
    ps.set_ground_plane_mode(ground_plane)

    frame_path = f"{save_path}/{frame_folder}"
    if save_base == True:
        ps_mesh = ps.register_surface_mesh("mesh", vertices, faces, enabled=True,
                                           edge_color=edge_color, edge_width=edge_width, material=material)
        ps.screenshot(f"{frame_path}/{name}.png")
        ps.remove_all_structures()
    Path(frame_path).mkdir(parents=True, exist_ok=True)
    # Convert 2D to 3D by appending Z-axis
    if vertices.shape[1] == 2:
        vertices = np.concatenate((vertices, np.zeros((len(vertices), 1))), axis=1)

    for i in range(len(angles)):
        rot = getRotMat(axis, angles[i])
        rot_verts = np.transpose(rot @ np.transpose(vertices))

        ps_mesh = ps.register_surface_mesh("mesh", rot_verts, faces, enabled=True,
                                           edge_color=edge_color, edge_width=edge_width, material=material)
        if scalars is not None:
            ps_mesh.add_scalar_quantity(f"scalar", scalars, defined_on=defined_on,
                                        cmap=cmap, enabled=True, vminmax=sminmax)
        if colors is not None:
            ps_mesh.add_color_quantity(f"color", colors, defined_on=defined_on,
                                       enabled=True)
        if highlight_faces is not None:
            # Create curve to highlight faces
            curve_v, new_f = trimMesh(rot_verts, faces[highlight_faces, :])
            curve_edges = []
            for face in new_f:
                curve_edges.extend(
                    [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]])
            curve_edges = np.array(curve_edges)
            ps_curve = ps.register_curve_network("curve", curve_v, curve_edges, color=highlight_color,
                                                 radius=highlight_radius)

        if cpos is None or clook is None:
            ps.reset_camera_to_home_view()
        else:
            ps.look_at(cpos, clook)

        if debug == True:
            ps.show()
        ps.screenshot(f"{frame_path}/{name}_{i}.png")
        ps.remove_all_structures()
    if save_video == True:
        import glob
        from PIL import Image
        fp_in = f"{frame_path}/{name}_*.png"
        fp_out = f"{save_path}/{name}.gif"
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=200, loop=0)