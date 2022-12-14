import random
import numpy as np
import k3d
from faker import Factory

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

def visualize_pointclouds_parts(point_clouds, labels, point_size=0.025, shift_vector=np.array([0,0,1])):
    """
    With batch_size = n, number of points = k.
    :param point_clouds: list, len n, each element is an array of shape (k, 3)
    :param labels: list, len n, each element is an array of shape (k, )
    """
    plot = k3d.plot(grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    part_colors = [0xff0000, 0x00ff00, 0x0000ff, 0xff00ff, 0xffff00, 0x00ffff]
    for i, point_cloud in enumerate(point_clouds):
        for label in np.unique(labels[i]):
            plt_points = k3d.points(positions=(i*shift_vector + point_cloud[label == labels[i]]).astype(np.float32), point_size=point_size, color=part_colors[label])
            plot += plt_points
    
    plt_points.shader = '3d'
    plot.display()

def visualize_pointclouds_parts_partglot(point_clouds, point_size=0.025, shift_vector=np.array([0,0,0]), names=None, part_colors=None, opacity=1):
    """
    With batch_size = n, number of points = k.
    :param point_clouds: list, len n, each element is an array of shape (k, 3)
    :param labels: list, len n, each element is an array of shape (k, )
    """
    plot = k3d.plot(grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))

    for i, point_cloud in enumerate(point_clouds):
        pc_name = names[i] if names else None
        pc_color = part_colors[i] if part_colors else get_rnd_color()
        plt_points = k3d.points(positions=(i*shift_vector + point_cloud).astype(np.float32), point_size=point_size, color=pc_color, name=pc_name, opacity=opacity)
        plot += plt_points
    
    plt_points.shader = '3d'
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

def visualize_meshes(meshes_list):
    plot = k3d.plot(name='points', grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    colors = [0xff0000, 0x00ff00, 0x0000ff, 0xff00ff, 0xffff00, 0x00ffff]
    for i, mesh in enumerate(meshes_list):
        plt_mesh = k3d.mesh(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32), color=colors[i%6])
        plot += plt_mesh
    plt_mesh.shader = '3d'
    plot.display()