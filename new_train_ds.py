#!/usr/bin/env python3
import os
import sys
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from argparse import ArgumentParser
import torch.nn as nn
import os
import threading
import deepspeed
import sys
import time
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import CLIP.clip as clip
from CLIP.clip import tokenize
from model import MagicLens
from new_data_utils_ds import build_happy_dataset_for_train, build_fiq_dataset_for_train, build_fiq_dataset_for_val, build_happy_dataset_for_val
import psutil
torch.cuda.empty_cache()
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer


def process_img(image_path: str, size: int) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((size, size), Image.BILINEAR)
        return np.array(img) / 255.0 

def print_memory_usage():
    while True:
        cpu_memory = psutil.virtual_memory().percent
        print(f"CPU memory usage: {cpu_memory}%")
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  
            gpu_memory_cached = torch.cuda.memory_reserved() / (1024 ** 3) 
            print(f"GPU memory usage: {gpu_memory:.2f} GB")
            print(f"GPU memory cached: {gpu_memory_cached:.2f} GB")
        else:
            print("No GPU available.")
        
        time.sleep(10)

def redirect_output_to_log(log_file):
    class LogRedirector:
        def __init__(self, log_file):
            self.log_file = log_file
            self.terminal = sys.stdout
            self.log = open(log_file, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = LogRedirector(log_file)

def contrastive_loss(r_q, r_t, r_t_prime, tau=0.07):
    sim_matrix = F.cosine_similarity(r_q.unsqueeze(1), r_t.unsqueeze(0), dim=-1)
    hard_nega_matrix = F.cosine_similarity(r_q.unsqueeze(1), r_t_prime.unsqueeze(0), dim=-1)

    sim_diag = torch.diagonal(sim_matrix)
    numerator = torch.exp(sim_diag / tau)
   
    part1 = torch.sum(torch.exp(sim_matrix / tau), dim=1)
    part2 = torch.sum(torch.exp(hard_nega_matrix / tau), dim=1)
    denominator = part1 + part2
    loss = -torch.log(numerator / denominator)
    return loss.mean()

def prepare_batch(batch, train_dataset):
    # print(batch.keys()) # dict_keys(['query', 'index'])

    qimages = torch.stack([qimage for qimage in batch['qimage']], dim=0)
    qtokens = torch.stack([qtokens for qtokens in batch['qtokens']], dim=0)

    timages, ttokens_list = [], []
    
    for q in batch['target_iid']:
        target_iid = q
        
        if isinstance(target_iid, list):
            target_iimages = [
                process_img(os.path.join(train_dataset.index_image_folder, f"{iid}.jpg"), 224)
                for iid in target_iid
            ]
            timages.append(torch.cat([torch.tensor(img) for img in target_iimages if img is not None]))

            target_tokens = np.array(tokenize("")).astype(np.float32)
            ttokens_list.append(torch.tensor(target_tokens))
        else:
            index_img_path = os.path.join(train_dataset.index_image_folder, f"{target_iid}.jpg")
            
            timage = process_img(index_img_path, 224) 
            
            if timage is not None:
                timages.append(torch.tensor(timage))

                token = np.array(tokenize("")).astype(np.float32)
                ttokens_list.append(torch.tensor(token))
            else:
                raise ValueError(f"Target image with ID {target_iid} not found.")
    
    assert len(timages) == len(ttokens_list), "Number of images and tokens must match."
    
    ttokens = torch.stack(ttokens_list)
    timages = torch.stack(timages)


    return qimages, qtokens, timages, ttokens

# def validate_model(model, val_loader, criterion):
#     model.eval().float()

#     total_loss = 0.0
#     num_batches = len(val_loader)
    
#     with torch.no_grad(): 
#         for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
#             qimages, qtokens, timages, ttokens = prepare_batch(batch, val_dataset)

#             # 将数据移到 GPU
#             qimages = qimages.to(device)
#             qtokens = qtokens.to(device)
#             timages = timages.to(device)
#             ttokens = ttokens.to(device)

#             qoutput = model({"ids": qtokens, "image": qimages})
#             query_embeddings = qoutput["multimodal_embed_norm"].to(device)

#             qhardoutput = model({"ids": ttokens, "image": qimages})
#             qhard_embeddings = qhardoutput["multimodal_embed_norm"].to(device)

#             toutput = model({"ids": ttokens, "image": timages})
#             target_embeddings = toutput["multimodal_embed_norm"].to(device)

#             loss = criterion(query_embeddings, target_embeddings, qhard_embeddings)
#             print(loss)
#             total_loss += loss.item()

#     avg_loss = total_loss / num_batches
#     print(f"Validation Loss: {avg_loss:.4f}")
    
#     return avg_loss

def train_model(model, train_loader, criterion, args):
    model.train().float()

    for epoch in range(args.epochs):
        total_loss = 0.0
        num_batches = len(train_loader)
        
        #load checkpoint
        _, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
        step = client_sd['step']

        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{cmd_args.epochs}")):
            qimages, qtokens, timages, ttokens = prepare_batch(batch, train_dataset)

            qoutput = model_engine({"ids": qtokens, "image": qimages})
            query_embeddings = qoutput["multimodal_embed_norm"]
            # print(query_embeddings.shape)

            qhardoutput = model_engine({"ids": ttokens, "image": qimages})
            qhard_embeddings = qhardoutput["multimodal_embed_norm"]
            # print(qhard_embeddings.shape)

            toutput = model_engine({"ids": ttokens, "image": timages})
            target_embeddings = toutput["multimodal_embed_norm"]

            loss = criterion(query_embeddings, target_embeddings, qhard_embeddings)
            # print("trian_loss: ",loss)

            model_engine.backward(loss)
            model_engine.step()

            if step % cmd_args.save_interval:
                client_sd['step'] = step
                ckpt_id = loss.item()
                model_engine.save_checkpoint('./train_deeptry', ckpt_id, client_sd = client_sd, save_latest=True)

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch + 1}/{args.epochs}], Training Loss: {avg_loss:.4f}")

if __name__ == "__main__":

    deepspeed.init_distributed()
    _local_rank = int(os.environ.get("LOCAL_RANK"))
    get_accelerator().set_device(_local_rank)

    # memory_thread = threading.Thread(target=print_memory_usage, daemon=True)
    # memory_thread.start()
    timestamp = int(time.time())
    timestamp = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H:%M:%S')
    output_dir = f"train_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, f"train_{timestamp}.log")
    redirect_output_to_log(log_file)

    parser = ArgumentParser(description='My training script.')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    parser.add_argument("--model_size", type=str, default="base", choices=["base", "large"], help="Model size.")
    parser.add_argument("--dataset", type=str, default="happy", choices=["fiq-dress", "fiq-shirt", "fiq-toptee", "circo", "dtin", "happy"], help="Dataset selection.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--log-interval",type=int,default=200,help="output logging information at a given interval")

    ds_config = {"train_batch_size": 100 ,"wall_clock_breakdown": False}

    # args = parser.parse_args()
    cmd_args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MagicLens(cmd_args.model_size).to(device)
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=cmd_args.lr)
    criterion = contrastive_loss

    if cmd_args.dataset.startswith("fiq"):
        subtask = cmd_args.dataset.split("-")[1]
        # train_dataset, train_loader = build_fiq_dataset_for_train(dataset_name=cmd_args.dataset)
        # val_dataset, val_loader = build_fiq_dataset_for_val(dataset_name=cmd_args.dataset)
        train_dataset, _ = build_fiq_dataset_for_train(dataset_name=cmd_args.dataset)
        val_dataset, _ = build_fiq_dataset_for_val(dataset_name=cmd_args.dataset)
    elif cmd_args.dataset in ["happy"]:
        # train_dataset, train_loader = build_happy_dataset_for_train(dataset_name=cmd_args.dataset)
        # val_dataset, val_loader = build_happy_dataset_for_val(dataset_name=cmd_args.dataset)
        train_dataset, _ = build_happy_dataset_for_train(dataset_name=cmd_args.dataset)
        val_dataset, _ = build_happy_dataset_for_val(dataset_name=cmd_args.dataset)
    else:
        raise NotImplementedError
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cmd_args.lr)

    model_engine, optimizer, train_loader, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     training_data = train_dataset,
                                                     config=ds_config,)
    data_iter = iter(train_loader)
    # Get the local device name (str) and local rank (int).
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank
    
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.8)


    train_model(model, train_loader, criterion, cmd_args)

    print("Training Done.")

