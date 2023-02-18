# TODO: cleanup
from pathlib import Path
from src.helper.preprocessing import remesh_per_part

shapenet_path = Path("/mnt/hdd/ShapeNetCore.v2")
mapping_path = Path("data/partglot_data/data_mapping.txt")
dest_path = Path("data/partglot_data")
items = mapping_path.read_text().splitlines()
for item in items:
    pg_id, synset_id, item_id = item.split("/")
    #remesh_per_part(shapenet_path.joinpath(f"{synset_id}/{item_id}/models/model_normalized.obj"), dest_path.joinpath(f"{synset_id}/{item_id}/mesh.obj"), remesh_iterations=5)
    remesh_per_part(dest_path.joinpath(f"{synset_id}/{item_id}/model_normalized_corrected_normals.obj"), dest_path.joinpath(f"{synset_id}/{item_id}/mesh_corrected_normals.obj"), remesh_iterations=5)