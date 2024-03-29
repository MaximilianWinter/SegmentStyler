from k3d.transform import process_transform_arguments
import numpy as np
import k3d
from faker import Factory
import matplotlib.pyplot as plt
import trimesh
import torch
import kaolin as kal
import torchvision.transforms as T

from src.data.mesh import Mesh
from src.submodels.render import Renderer

fake = Factory.create()
get_rnd_color = lambda: int(fake.hex_color().replace('#', '0x'), 16)
get_colors = lambda n: [get_rnd_color() for _ in range(n)]

def visualize_pointcloud(point_cloud, point_size=0.025, flip_axes=False, name='point_cloud'):
    plot = k3d.plot(name=name, grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if flip_axes:
        point_cloud[:, 2] = point_cloud[:, 2] * -1
        point_cloud[:, [0, 1, 2]] = point_cloud[:, [0, 2, 1]]
    plt_points = k3d.points(positions=point_cloud.astype(np.float32), point_size=point_size, color=0xd0d0d0)
    plot += plt_points
    plt_points.shader = '3d'
    plot.display()

def visualize_pointclouds_parts(point_clouds, labels, point_size=0.025, shift_vector=np.array([0,0,1]), part_colors = None):
    """
    With batch_size = n, number of points = k.
    :param point_clouds: list, len n, each element is an array of shape (k, 3)
    :param labels: list, len n, each element is an array of shape (k, )
    """
    plot = k3d.plot(grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if part_colors is None:
        part_colors = [0xff0000, 0x00ff00, 0x0000ff, 0xff00ff, 0xffff00, 0x00ffff]
    for i, point_cloud in enumerate(point_clouds):
        for label in np.unique(labels[i]):
            plt_points = k3d.points(positions=(i*shift_vector + point_cloud[label == labels[i]]).astype(np.float32), point_size=point_size, color=part_colors[label])
            plot += plt_points
    
    plt_points.shader = '3d'
    plot.display()

def visualize_pointclouds(point_clouds, point_size=0.025, shift_vector=np.array([0,0,1])):
    """
    With batch_size = n, number of points = k.
    :param point_clouds: list, len n, each element is an array of shape (k, 3)
    :param labels: list, len n, each element is an array of shape (k, )
    """
    plot = k3d.plot(grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    part_colors = [0xff0000, 0x00ff00, 0x0000ff, 0xff00ff, 0xffff00, 0x00ffff]
    for i, point_cloud in enumerate(point_clouds):
        plt_points = k3d.points((point_cloud).astype(np.float32), point_size=point_size, color=part_colors[i])
        plot += plt_points
    
    plt_points.shader = '3d'
    plot.display()

def visualize_pointclouds_parts_partglot(point_clouds, point_size=0.025, shift_vector=np.array([0,0,0]), names=None, part_colors=None, opacity=1, dump_screenshots=True):
    """
    With batch_size = n, number of points = k.
    :param point_clouds: list, len n, each element is an array of shape (k, 3)
    :param labels: list, len n, each element is an array of shape (k, )
    """
    print('Visualization rendering started...')
    
    
    plot = k3d.plot(grid_visible=False)

    for i, point_cloud in enumerate(point_clouds):
        pc_name = names[i] if names else None
        pc_color = part_colors[i] if part_colors else get_rnd_color()
        plt_points = k3d.points(positions=(i*shift_vector + point_cloud).astype(np.float32), point_size=point_size, color=pc_color, name=pc_name, opacity=opacity, grid_auto_fit=True)
        plt_points = process_transform_arguments(plt_points, rotation=[np.pi, -2 * np.pi / 6, -4.5 * np.pi / 6, -5.5*np.pi/6])
        plot += plt_points
    

    plot.display()
    

def visualize_mesh(vertices, faces, flip_axes=False):
    plot = k3d.plot(name='points', grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if flip_axes:
        vertices[:, 2] = vertices[:, 2] * -1
        vertices[:, [0, 1, 2]] = vertices[:, [0, 2, 1]]
    plt_mesh = k3d.mesh(vertices.astype(np.float32), faces.astype(np.uint32), color=0xd0d0d0)
    plot += plt_mesh
    plt_mesh.shader = '3d'
    plot.display()

def visualize_meshes(meshes_list, process_trafo_args=True):
    plot = k3d.plot(name='points', grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    colors = [0xff0000, 0x00ff00, 0x0000ff, 0xff00ff, 0xffff00, 0x00ffff]
    for i, mesh in enumerate(meshes_list):
        if isinstance(mesh, dict):
            vertices, faces = mesh['vertices'].astype(np.float32), mesh['faces'].astype(np.uint32)
        else:
            vertices, faces = mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32)
        plt_mesh = k3d.mesh(vertices, faces, color=colors[i%6])
        if process_trafo_args == True:
            plt_mesh = process_transform_arguments(plt_mesh, rotation=[np.pi, -2 * np.pi / 6, -4.5 * np.pi / 6, -5.5*np.pi/6])
        plot += plt_mesh
    plt_mesh.shader = '3d'
    plot.display()
    
    
    
def plot_pointclouds(pointcloud_list, psize=0.01, elev=45, azim=-70, start_idx=0):
    fig = plt.figure(figsize=(15,6))
    num_pointclouds = len(pointcloud_list)
    rows = int(num_pointclouds**0.5)
    cols = int(num_pointclouds / rows)
    if rows * cols < num_pointclouds:
        cols += 1

    for i, pointcloud in enumerate(pointcloud_list):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        ax.scatter(pointcloud[:,0], pointcloud[:,1], pointcloud[:,2], s=psize)
        ax.set_axis_off()
        ax.set_title(f'i={start_idx+i}')
        ax.view_init(elev=elev, azim=azim)


    plt.show()

class GifCreator:

    def __init__(self, res=512):
        """Small class for exporting animated GIFs from objs.
        :param res: int, output resolution
        """
        self.renderer = Renderer(dim=(res, res))
        self.transform = T.ToPILImage()

    def export_gif(self, base_path, num_views=24):
        """
        Expects the mesh to be stored in base_path/final_mesh.obj.
        Stores animated GIF in base_path.
        :param base_path: pathlib.Path object
        :param num_views: int, number of views to be rendered
        """
        tri_mesh = trimesh.load(base_path.joinpath("final_mesh.obj"))
        mesh = Mesh(str(base_path.joinpath("final_mesh.obj")), use_trimesh=True)
        rgb = (
            torch.tensor(tri_mesh.visual.vertex_colors[:, :3] / 255.0)
            .float()
            .to(mesh.vertices.device)
        )
        mesh.face_attributes = kal.ops.mesh.index_vertices_by_faces(
            rgb.unsqueeze(0), mesh.faces
        )
        mesh.vertex_colors = rgb

        azims = torch.linspace(0, 2*np.pi, num_views)
        elevs = torch.tensor([torch.pi/6 for i in range(len(azims))])

        imgs = self.renderer.render_given_front_views(mesh, elevs, azims, lighting=False, show=False, background=1)

        pil_img = self.transform(imgs[0])
        other_pil_imgs = [self.transform(img) for img in imgs[1:]]

        try:
            save_name = int(str(base_path).split("_")[-1]) # use version number as file name
        except ValueError:
            save_name = "out"

        pil_img.save(base_path.joinpath(f"{save_name}.gif"), save_all=True, append_images=other_pil_imgs, duration=100, loop=0)