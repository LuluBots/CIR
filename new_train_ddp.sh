CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --master_port 29501 --nproc_per_node 4 --nnodes 1 new_train_ddp.py 
