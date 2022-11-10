def path_to_image_html(path, param='height', val=150):
    return f'<img src="{str(path)}" {param}="{val}" >'