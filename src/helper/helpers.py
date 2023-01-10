from six.moves import cPickle


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
