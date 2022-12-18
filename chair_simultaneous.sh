python main.py \
--obj_path data/chair_testmesh.obj --mask_path data/chair_testmesh_mask.jsonc --sigma 5.0 --geoloss --clamp tanh --n_normaugs 4 --n_augs 1 --normmincrop 0.1 --normmaxcrop 0.1 --frontview --frontview_std 4 --clipavg view --lr_decay 0.9 --clamp tanh --normclamp tanh --maxcrop 1.0 --save_render --seed 78942387 --save_render --frontview_center 1.96349 0.6283 --colordepth 2 --normdepth 2 --n_iter 1500 --reg_lambda 1e-2 --learning_rate 0.005 --output_dir multi_prompt --experiment_group multi_prompt_simultaneous --prompt "a chair with a tiger seat" --prompt "a chair with a green checked pattern back" --loss_name penalizing_loss --use_previous_prediction --width 512 --depth 2 --model_name Text2MeshExtended