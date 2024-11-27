# model configs
resolution=3
laplace_loss_min_beta=-2.0
pred_confidence_input=1
multi_resolution_learnable_gradients_weights="learnable"
optim_layer_input_clamp=1.0
depth_activation_format='exp'

# training configs
batch_size=6
lr=0.001
train_depth_noise=0.0~0.05
train_depth_pattern="0.5*100~2000+0.25*sift+0.25*velodyne"
mixed_dataset_total_length=125000

python main.py \
    --train_data_name Hypersim+IRS+VKITTI+TartanAir+BlendedMVS --mixed_dataset_total_length $mixed_dataset_total_length \
    --train_depth_pattern $train_depth_pattern --train_depth_noise $train_depth_noise \
    --random_rot_deg 0.0 \
    --resize_height 480 --resize_width 640 --patch_height 480 --patch_width 640 \
    --max_depth 10.0 --data_normalize_median 1 \
    --val_data_name NYU_FULL_RES --val_depth_pattern 2000 \
    --num_resolution $resolution --multi_resolution_learnable_gradients_weights $multi_resolution_learnable_gradients_weights \
    --gpus 0,1,2,3,4,5,6,7,8,9 --multiprocessing \
    --lr $lr --batch_size $batch_size --milestones 36 48 56 64 72 --epochs 72 \
    --loss 0.5*SeqLaplace+1.0*SeqGradL1+1.0*ConfInput+1.0*SeqL1+2.0*GradMatchingScale --laplace_loss_min_beta $laplace_loss_min_beta \
    --intermediate_loss_weight 1.0 \
    --backbone_mode rgbd --pred_confidence_input $pred_confidence_input \
    --GRU_iters 1 --optim_layer_input_clamp $optim_layer_input_clamp --depth_activation_format $depth_activation_format \
    --whiten_sparse_depths 1 --gru_internal_whiten_method median \
    --log_dir ../experiments/ \
    --save "train_10gpu_72epochs" \
    --save_full