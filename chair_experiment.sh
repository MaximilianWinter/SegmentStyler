python main.py \
--obj_path data/chair_testmesh.obj \
--mask_path data/chair_testmesh_mask.jsonc \
--output_dir tiger_seat \
--prompt a chair with a tiger seat \
--sigma 5.0 \
--geoloss \
--clamp tanh \
--n_normaugs 4 \
--n_augs 1 \
--normmincrop 0.1 \
--normmaxcrop 0.1 \
--colordepth 2 \
--normdepth 2 \
--frontview \
--frontview_std 4 \
--clipavg view \
--lr_decay 0.9 \
--clamp tanh \
--normclamp tanh \
--maxcrop 1.0 \
--save_render \
--seed 78942387 \
--save_render \
--n_iter 500 \
--learning_rate 0.0005 \
--normal_learning_rate 0.0005 \
--frontview_center 1.96349 0.6283