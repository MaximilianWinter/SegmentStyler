for depth in 2 4 8 16
do
    for width in 64 128 256 512
    do
        for lr in 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2
        do
            python main.py \
            --obj_path data/chair_testmesh.obj \
            --mask_path data/chair_testmesh_mask.jsonc \
            --reg_lambda 1e-2 \
            --output_dir tiger_seat_hparam \
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
            --n_iter 1000 \
            --learning_rate $lr \
            --frontview_center 1.96349 0.6283 \
            --loss_name penalizing_loss \
            --depth $depth \
            --width $width
        done
    done
done