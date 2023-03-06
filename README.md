# SegmentStyler [[Project's Report](/data/report/ReportSegmentStyler.pdf)]
## Part-aware language-based mesh texture editing

SegmentStyler is a framework for part-aware mesh texture editing that uses natural language and allows for separate customization of different mesh segments based on multiple text prompts, resulting in a coherent texture with the text input.

<div align="center">

| <img src="/data/img/example_chair_1.gif" width="130"> | <img src="/data/img/example_chair_2.gif" width="130"> | <img src="/data/img/example_chair_3.gif" width="130"> | <img src="/data/img/example_chair_4.gif" width="130"> | 
| :--: | :--: | :--: | :--: |
| <small align="center">rainbow pattern **arm**<br>glowing iron **leg**<br>guy fawkes logo **back**<br>camo pattern **seat**</small> | <small align="center">persian carpet **arm**<br>swiss cheese **leg**<br>desert sand **back**<br>fleur-de-lis pattern **seat**</small> | <small align="center">watermelon pattern **arm**<br>red leather **leg**<br>graffiti **back**<br>colorful crochet **seat**</small> | <small align="center">snowflake pattern **arm**<br>metallic **leg**<br>honeycomb **back**<br>wooden **seat**</small> |

</div>

### Note

> This project was implemented during a Guided Research practical course at [TUM](https://www.tum.de/) in the Winter Semester of 2022/2023 by [Maximilian Winter](mailto:maximilian96.winter@tum.de) and [Murilo Bellatini](mailto:bellatini@in.tum.de). For more information, please refer to the project's report, which can be found [here](/data/report/ReportSegmentStyler.pdf).

## Getting Started

### Installation

1. Run `setup.sh`.

```bash
sh ./setup.sh
```

2. Setup PartGlot 

    1. Download data and weights [here](https://drive.google.com/drive/folders/1jvPclGP5Dg0653wrMvN8WX9am7txZJu8).
    2. Add data files and weights to `data/PartGlotData` folder.

3. Setup Data (Note: this step can be skipped if you only want to try out the pipeline with the one shape that's already given in `data/`)
    1. Download ShapeNetCore.v2 [here](https://shapenet.org/)
    2. Download the part annotations [here](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip)
    3. Start pre-processing (remeshing and fixing normals) by running 
    ```
    python src/helper/get_new_normals_from_sdf.py
    ```
    For the pre-processing, make sure to have the following packages installed
    - trimesh
    - mesh_to_sdf
    - skimage
    To avoid conflicts, it is recommended to install them (especially the last two) in a separate environment.
    
5. Activate environment.

```bash
conda activate dl3d
```

4. Confirm Weights & Biases installation. More info: [click here](https://wandb.ai/quickstart/pytorch).

```bash
pip install wandb
wandb login
```

5. Enjoy ;)


### Run examples

To see an example, you can run
```
bash example_run.sh
```

To gain more insights, you can have a look at
- `evaluation_baseline.sh`
- `ablation_study_b_to_g.sh`
which were used for performing our ablation study.

## Reference Projects

1. [Text2Mesh](https://github.com/threedle/text2mesh)
2. [PartGlot](https://github.com/63days/PartGlot)