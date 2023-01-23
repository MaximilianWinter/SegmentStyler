from six.moves import cPickle
import numpy as np


def path_to_image_html(path, param='height', val=150):
    return f'<img src="{str(path)}" {param}="{val}" >'

def pickle_data(file_name, *args):
    """Using (c)Pickle to save multiple python objects in a single file."""
    out_file = open(file_name, "wb")
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
    :return: a generator over the un-pickled items.
    Note, about implementing the python2_to_3 see
        https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """

    in_file = open(file_name, "rb")
    if python2_to_3:
        size = cPickle.load(in_file, encoding="latin1")
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding="latin1")
        else:
            yield cPickle.load(in_file)
    in_file.close()

def read_obj(path):
    lines = path.read_text().splitlines()
    vertices = []
    vertex_normals = []
    faces = []
    face_normals = []

    for line in lines:
        elements = line.split(" ")
        
        if elements[0] == "v":
            vertices.append([float(val) for val in elements[1:4]])
        elif elements[0] == "vn":
            vertex_normals.append([float(val) for val in elements[1:4]])
        elif elements[0] == "f":
            face_list = []
            face_normal_index_list = []
            cleaned_elements = [val for val in elements if val != ""]
            for val in cleaned_elements[1:4]:
                face, _, face_normal_index = val.split("/")
                face_list.append(int(face)-1)
                face_normal_index_list.append(int(face_normal_index)-1)
            faces.append(face_list)
            face_normals.append(face_normal_index_list)
    
    return np.array(vertices), np.array(vertex_normals), np.array(faces), np.array(face_normals)

def write_obj(path, v, vn, f, fn):
    with open(path, "w") as fp:
        for vertex in v:
            fp.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

        for val in vn:
            fp.write(f"vn {val[0]:.6f} {val[1]:.6f} {val[2]:.6f}\n")

        for face, face_norm in zip(f, fn):
            face += 1
            face_norm +=1
            fp.write(f"f {face[0]}//{face_norm[0]} {face[1]}//{face_norm[1]} {face[2]}//{face_norm[2]}\n")
        
