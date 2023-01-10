"""
All relevant paths stored in constant variables
"""

import pathlib as pl

# local paths
root_path = pl.Path(__file__).parents[2]
LOCAL_DATA_PATH           = root_path / 'data'
LOCAL_RAW_DATA_PATH       = LOCAL_DATA_PATH / 'raw'
LOCAL_INTERIM_DATA_PATH   = LOCAL_DATA_PATH / 'interim'
LOCAL_PROCESSED_DATA_PATH = LOCAL_DATA_PATH / 'processed'
LOCAL_MODELS_PATH         = root_path / 'models'
LOCAL_CREDENTIALS_PATH    = root_path / 'credentials'
LOCAL_LOGGING_PATH        = root_path / 'logging/log.log'
LOCAL_CONFIG_PATH         = root_path / 'src/config'

NOTEBOOKS_PATH = root_path / 'notebooks'
BASELINES_PATH = root_path / 'Baselines'

# global paths
GLOBAL_DATA_PATH          = pl.Path('/mnt/hdd')


pg2shapenet = {
    40: {'mesh': "/mnt/hdd/ShapeNetCore.v2/03001627/88c39cf1485b497bfbb8cbddab1c2002/models/model_normalized.obj",
         'labels': "/mnt/hdd/shapenetcore_partanno_segmentation_benchmark_v0/03001627/points_label/88c39cf1485b497bfbb8cbddab1c2002.seg",
         'pc': '/mnt/hdd/shapenetcore_partanno_segmentation_benchmark_v0/03001627/points/88c39cf1485b497bfbb8cbddab1c2002.pts'},
    55: {'mesh': "/mnt/hdd/ShapeNetCore.v2/03001627/c1b312919af633f8f51f77a6d7299806/models/model_normalized.obj",
         'labels': "/mnt/hdd/shapenetcore_partanno_segmentation_benchmark_v0/03001627/points_label/c1b312919af633f8f51f77a6d7299806.seg",
         'pc': '/mnt/hdd/shapenetcore_partanno_segmentation_benchmark_v0/03001627/points/c1b312919af633f8f51f77a6d7299806.pts'},
    70: {'mesh': "/mnt/hdd/ShapeNetCore.v2/03001627/47eff1e5c09cec7435836c728d324152/models/model_normalized.obj",
         'labels': "/mnt/hdd/shapenetcore_partanno_segmentation_benchmark_v0/03001627/points_label/47eff1e5c09cec7435836c728d324152.seg",
         'pc': '/mnt/hdd/shapenetcore_partanno_segmentation_benchmark_v0/03001627/points/47eff1e5c09cec7435836c728d324152.pts'}
}