export CUDA_VISIBLE_DEVICES=0;
python test.py --dataroot /home/ht1/Radboud_selectiongan/ --name facial_exp --model BiGraphGAN --phase test --dataset_mode aligned --norm batch --batchSize 1 --resize_or_crop no --gpu_ids 0 --BP_input_nc 3 --no_flip --which_model_netG Graph --checkpoints_dir ./checkpoints --which_epoch 200 --results_dir ./results/ --display_id 0;

