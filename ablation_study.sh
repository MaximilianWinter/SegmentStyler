while read p1; do
    read p2
    read p3
    read p4
    while read sample; do
        echo "$p1"
        echo "$p2"
        echo "$p3"
        echo "$p4"
        echo "$sample"

        python main.py \
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
        --n_iter 1500 \
        --learning_rate 0.005 \
        --output_dir evaluation_b \
        --experiment_group evaluation_b \
        --width 32 \
        --depth 2 \
        --model_name Text2MeshMultiMLP \
        --loss_name multi_mlp_loss \
        --dataset PartGlotData \
        --sample $sample \
        --prompt "$p1" \
        --prompt "$p2" \
        --prompt "$p3" \
        --prompt "$p4"

        python main.py \
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
        --n_iter 1500 \
        --learning_rate 0.005 \
        --output_dir evaluation_c \
        --experiment_group evaluation_c \
        --width 32 \
        --depth 2 \
        --model_name Text2MeshMultiMLP \
        --loss_name multi_mlp_loss \
        --do_backward_masking \
        --dataset PartGlotData \
        --sample $sample \
        --prompt "$p1" \
        --prompt "$p2" \
        --prompt "$p3" \
        --prompt "$p4"

        python main.py \
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
        --n_iter 1500 \
        --learning_rate 0.005 \
        --output_dir evaluation_e \
        --experiment_group evaluation_e \
        --width 32 \
        --depth 2 \
        --model_name Text2MeshMultiMLP \
        --loss_name multi_mlp_loss \
        --do_backward_masking \
        --biased_views \
        --gaussian_blending \
        --final_gaussian_blending \
        --dataset PartGlotData \
        --sample $sample \
        --prompt "$p1" \
        --prompt "$p2" \
        --prompt "$p3" \
        --prompt "$p4"

        python main.py \
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
        --n_iter 1500 \
        --learning_rate 0.005 \
        --output_dir evaluation_f \
        --experiment_group evaluation_f \
        --width 32 \
        --depth 2 \
        --model_name Text2MeshMultiMLP \
        --loss_name multi_mlp_loss \
        --do_backward_masking \
        --biased_views \
        --use_gt_masks \
        --dataset PartGlotData \
        --sample $sample \
        --prompt "$p1" \
        --prompt "$p2" \
        --prompt "$p3" \
        --prompt "$p4"
    done <data/samples.txt
done <data/new_uncombined_sentences.txt
