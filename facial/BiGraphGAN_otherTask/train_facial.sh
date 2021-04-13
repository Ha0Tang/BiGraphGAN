export CUDA_VISIBLE_DEVICES=0,1,2;
python train.py --dataroot /home/ht1/Radboud_selectiongan/ --name facial_exp --model BiGraphGAN --lambda_GAN 5 --lambda_A 10  --lambda_B 10 --dataset_mode aligned --no_lsgan --n_layers 3 --norm batch --batchSize 24 --resize_or_crop no --gpu_ids 0,1,2 --BP_input_nc 3 --no_flip --which_model_netG Graph --niter 100 --niter_decay 100 --checkpoints_dir ./checkpoints --L1_type l1_plus_perL1 --n_layers_D 3 --with_D_PP 1 --with_D_PB 1  --display_id 0 --save_epoch_freq 50
#--continue_train --which_epoch 640 --epoch_count 641
