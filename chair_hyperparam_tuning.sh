for lambda in 0.001 0.0025 0.005 0.0075 0.01 0.025 0.05 0.075 0.1
do
    python main.py \
    --obj_path data/chair_testmesh.obj \
    --mask_path data/chair_testmesh_mask.jsonc \
    --sigma 5.0 \
    --geoloss \
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
    --frontview_center 1.96349 0.6283 \
    --colordepth 2 \
    --normdepth 2 \
    --n_iter 1500 \
    --reg_lambda $lambda \
    --learning_rate 0.005 \
    --output_dir tuning_lambda \
    --experiment_group tuning_lambda \
    --prompt "a chair with a tiger seat" \
    --loss_name penalizing_loss \
    --width 512 \
    --depth 2 \
    --round_renderer_gradients
done