#!/usr/bin/env python3
import os
# os.environ['TORCH_CUDA_ARCH_LIST']
import sys
import time
from datetime import datetime
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

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
from new_data_utils_ds import build_happy_dataset_for_train, build_fiq_dataset_for_train, build_happy_dataset_for_val
import psutil
torch.cuda.empty_cache()
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from deepspeed.accelerator import get_accelerator
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

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


def custom_collate_fn(batch):
    # batch 是包含多个样本的列表，每个样本是一个字典，包含 'query' 和 'index'
    queries = [sample['query'] for sample in batch]
    indexes = [sample['index'] for sample in batch]

    # 获取 query 中所有字段
    qid = [q.qid for q in queries]
    # print(type(q.qimage)for q in queries)
    qimage = torch.stack([torch.tensor(q.qimage).float() for q in queries])  # 将图像转换为张量
    qtokens = [q.qtokens for q in queries]  # tokens 通常是 numpy 数组或列表，确保它们是相同大小
    target_iid = [q.target_iid for q in queries]
    retrieved_iid = [q.retrieved_iids for q in queries]
    retrieved_scores = [q.retrieved_scores for q in queries]

    # 如果 tokens 的维度不统一，可以选择对其进行填充（padding），这里假设它们已经统一
    # qtokens = torch.tensor(np.array(qtokens))  # 假设 tokens 是二维数组
    qtokens = pad_sequence(qtokens, batch_first=True, padding_value=0)  # 默认填充值是0

    # 如果 retrieved_iids 或 retrieved_scores 的长度不相同，可能需要做额外的处理，比如填充
    retrieved_iid = [torch.tensor(i) for i in retrieved_iid]  # 可以选择将其转换为张量
    retrieved_scores = [torch.tensor(s) for s in retrieved_scores]  # 同样转换为张量

    # 对 index 示例进行处理（这里我们只关心 image 和 tokens）
    iid = [i.iid for i in indexes]
    iimage = torch.stack([torch.tensor(i.iimage).float() for i in indexes])
    itokens = [i.itokens for i in indexes]
    # itokens = torch.tensor(np.array(itokens))  # 假设它们是二维数组
    itokens = pad_sequence(itokens, batch_first=True, padding_value=0)

    # 返回处理后的数据
    return {
        'qid': qid,
        'qimage': qimage,
        'qtokens': qtokens,
        'target_iid': target_iid,
        'retrieved_iid': retrieved_iid,
        'retrieved_scores': retrieved_scores,
        'iid': iid,
        'iimage': iimage,
        'itokens': itokens
    }


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

    qimages = torch.stack([qimage for qimage in batch['qimage']], dim=0)
    qtokens = torch.stack([qtokens for qtokens in batch['qtokens']], dim=0)

    timages, ttokens_list = [], []
    
    for q in batch['target_iid']:
        target_iid = q
        
        if isinstance(target_iid, list):
            target_iimages = [
                process_img(os.path.join(train_dataset.index_image_folder, f"{iid}.png"), 224)
                for iid in target_iid
            ]
            timages.append(torch.cat([torch.tensor(img) for img in target_iimages if img is not None]))

            target_tokens = np.array(tokenize("")).astype(np.float32)
            ttokens_list.append(torch.tensor(target_tokens))
        else:
            index_img_path = os.path.join(train_dataset.index_image_folder, f"{target_iid}.png")
            
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
        # _, client_sd = model_engine.load_checkpoint(args.load_dir, args.ckpt_id)
        # step = client_sd['step']

        for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{args.epochs}")):
            qimages, qtokens, timages, ttokens = prepare_batch(batch, train_dataset)

            qimages = qimages.to(device)
            qtokens = qtokens.to(device)
            timages = timages.to(device)
            ttokens = ttokens.to(device)

            qoutput = model_engine({"ids": qtokens, "image": qimages})
            query_embeddings = qoutput["multimodal_embed_norm"].to(device)
            # print(query_embeddings.shape)

            qhardoutput = model_engine({"ids": ttokens, "image": qimages})
            qhard_embeddings = qhardoutput["multimodal_embed_norm"].to(device)
            # print(qhard_embeddings.shape)

            toutput = model_engine({"ids": ttokens, "image": timages})
            target_embeddings = toutput["multimodal_embed_norm"].to(device)

            loss = criterion(query_embeddings, target_embeddings, qhard_embeddings)
            # print("trian_loss: ",loss)

            model_engine.backward(loss)
            model_engine.step()

            # if step % args.log_interval:
            #     # client_sd['step'] = step
            #     ckpt_id = loss.item()
            #     model_engine.save_checkpoint('./train_deeptry', step, save_latest=True)

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
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument("--model_size", type=str, default="base", choices=["base", "large"], help="Model size.")
    parser.add_argument("--dataset", type=str, default="fiq-dress", choices=["fiq-dress", "fiq-shirt", "fiq-toptee", "circo", "dtin", "happy"], help="Dataset selection.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    # parser.add_argument("--bacth_size", type=int, default=100)
    parser.add_argument("--log_interval",type=int,default=2000, help="output logging information at a given interval")
    args = parser.parse_args()

    ds_config = {
                # "train_batch_size": args.batch_size,
                "train_micro_batch_size_per_gpu":200,
                "wall_clock_breakdown": False,
                "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.001,
                    "betas": [0.8, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7,
                }}}


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MagicLens(args.model_size).to(device)
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()
    criterion = contrastive_loss

    if args.dataset.startswith("fiq"):
        subtask = args.dataset.split("-")[1]
        train_dataset = build_fiq_dataset_for_train(dataset_name=args.dataset)
        train_loader = DataLoader(train_dataset, sampler = DistributedSampler(train_dataset, shuffle=True), num_workers=4, collate_fn=custom_collate_fn)  
    # elif args.dataset in ["happy"]:
    #     train_dataset = build_happy_dataset_for_train(dataset_name=args.dataset)
    #     val_dataset = build_happy_dataset_for_val(dataset_name=args.dataset)
    else:
        raise NotImplementedError
    

    # data_iter = iter(train_loader)

    model_engine, optimizer, _ , _ = deepspeed.initialize(args=args,
                                                     model=model,
                                                     model_parameters=model.parameters(),
                                                     config=ds_config)
    
    # Get the local device name (str) and local rank (int).
    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

    train_model(model, train_loader, criterion, args)

    print("Training Done.")

