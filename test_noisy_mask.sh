python main.py \
--obj_path data/chair_testmesh.obj \
--mask_path data/chair_testmesh_mask.jsonc \
--sigma 5.0 \
--clamp tanh \
--n_normaugs 4 \
--n_augs 1 \
--normmincrop 0.1 \
--normmaxcrop 0.1 \
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
--frontview_center 5.4 -0.5 \
--colordepth 2 \
--normdepth 2 \
--n_iter 1 \
--learning_rate 0.005 \
--output_dir noisy_masks \
--experiment_group noisy_masks \
--prompt "a tiger pattern seat" \
--prompt "a nightsky pattern back" \
--prompt "rainbow legs" \
--width 32 \
--depth 2 \
--model_name Text2MeshMultiMLP \
--loss_name multi_mlp_loss \
--do_backward_masking \
--noisy_masks
#--weights_path "logs/2022-12-19/noisy_masks/version_19/final_mlp.pt"
#--final_gaussian_blending 
#--optimize_displacement
#--round_renderer_gradients
#--prompt "a tiger pattern back" \
#--prompt "rainbow legs" \
#--final_gaussian_blending \