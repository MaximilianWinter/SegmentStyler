while read sample; do
    while read p; do
        echo "$p"
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
        --output_dir evaluation_a \
        --experiment_group evaluation_a \
        --width 32 \
        --depth 2 \
        --model_name Text2MeshOriginal \
        --loss_name default_loss \
        --dataset PartGlotData \
        --sample $sample \
        --prompt "$p" 
    done <data/combined_sentences.txt
done <data/samples.txt