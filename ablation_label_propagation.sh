for version in {0..250}
do
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
    --n_iter 500 \
    --learning_rate 0.005 \
    --output_dir evaluation_g \
    --experiment_group evaluation_g \
    --width 32 \
    --depth 2 \
    --model_name Text2MeshLabelPropagation \
    --loss_name label_propagation_loss \
    --do_backward_masking \
    --biased_views \
    --gaussian_blending \
    --final_gaussian_blending \
    --dataset PartGlotData \
    --eval_version $version \
    --weights_path "logs/evaluation_dir/evaluation_d/version_$version/final_mlp.pt" \
    "--optimize_learned_labels"
done