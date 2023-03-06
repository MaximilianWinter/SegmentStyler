# SegmentStyler
## Part-aware language-based mesh texture editing
SegmentStyler is a framework for part-aware mesh texture editing that uses natural language and allows for separate customization of different mesh segments based on multiple text prompts, resulting in a coherent texture with the text input.


| rainbow pattern arm<br>glowing iron leg<br>guy fawkes logo back<br>camo pattern seat | persian carpet arm<br>swiss cheese leg<br>desert sand back<br>fleur-de-lis pattern seat  | watermelon pattern arm<br>red leather leg<br>graffiti back<br>colorful crochet seat | snowflake pattern arm<br>metallic leg<br>honeycomb back<br>wooden seat |
| ---- | ---- | ---- | --- |
| ![Chair 1](/data/img/example_chair_1.gif) | ![Chair 12](/data/img/example_chair_2.gif) | ![Chair 3](/data/img/example_chair_3.gif) | ![Chair 4](/data/img/example_chair_4.gif) | 

### Purpose of this repo

> Explore language guided models for mesh editing and generation with the goal of achieving local details. 

### How to run

1. Run `setup.sh`.

```bash
sh ./setup.sh
```

2. Setup PartGlot 

    1. Download data and weights [here](https://drive.google.com/drive/folders/1jvPclGP5Dg0653wrMvN8WX9am7txZJu8).
    2. Add data files and weights to `data/PartGlotData` folder.

3. Setup Data (this step can be skipped if you only want to try out the pipeline with the one shape that's already given in data/)
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
    
3. Activate environment.

```bash
conda activate dl3d
```

4. Confirm Weights & Biases installation. More info: [click here](https://wandb.ai/quickstart/pytorch).

```bash
pip install wandb
wandb login
```

5. Enjoy ;)

To see an example, you can run
```
bash example_run.sh
```

To gain more insights, you can have a look at
- evaluation_baseline.sh
- ablation_study_b_to_g.sh
which were used for performing our ablation study.
