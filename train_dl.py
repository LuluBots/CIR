
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
from data_dl import build_happy_dataset_for_train, build_fiq_dataset_for_train, build_fiq_dataset_for_val, build_happy_dataset_for_val
import psutil
torch.cuda.empty_cache()
from PIL import Image

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
        
        time.sleep(60)

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
    # print(r_q.shape,r_t.shape, r_t_prime.shape)
    # unsqueeze(1) (n, 1, d)  unsqueeze(0) (1, n, d)   dim=-1 在最后一维（特征维度）上计算余弦相似度 得到n*n的(r_q[i], r_t[j])
    sim_matrix = F.cosine_similarity(r_q.unsqueeze(1), r_t.unsqueeze(0), dim=-1)
    # hard_nega即“难负样本”，encode (query_image, null_token) to get r_t_prime
    hard_nega_matrix = F.cosine_similarity(r_q.unsqueeze(1), r_t_prime.unsqueeze(0), dim=-1)

    # 返回包含矩阵主对角线元素的一维张量，r_q[i] 和 r_t[j] 之间的余弦相似度
    sim_diag = torch.diagonal(sim_matrix)
    numerator = torch.exp(sim_diag / tau)
   
    # dim=1即按行求和
    part1 = torch.sum(torch.exp(sim_matrix / tau), dim=1)
    part2 = torch.sum(torch.exp(hard_nega_matrix / tau), dim=1)
    denominator = part1 + part2
    loss = -torch.log(numerator / denominator)
    # print(loss)
    # print(loss.mean())
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

def train_model(model, train_loader, optimizer, criterion, args):
    model.train().float()
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        total_loss = 0.0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{args.epochs}")):

            optimizer.zero_grad()

            qimages, qtokens, timages, ttokens = prepare_batch(batch, train_dataset)

            # 将数据移到 GPU
            qimages = qimages.to(device)
            qtokens = qtokens.to(device)
            timages = timages.to(device)
            ttokens = ttokens.to(device)

            qoutput = model({"ids": qtokens, "image": qimages})
            query_embeddings = qoutput["multimodal_embed_norm"].to(device)
            # print(query_embeddings.shape)

            qhardoutput = model({"ids": ttokens, "image": qimages})
            qhard_embeddings = qhardoutput["multimodal_embed_norm"].to(device)
            # print(qhard_embeddings.shape)

            toutput = model({"ids": ttokens, "image": timages})
            target_embeddings = toutput["multimodal_embed_norm"].to(device)

            loss = criterion(query_embeddings, target_embeddings, qhard_embeddings)
            # print("trian_loss: ",loss)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch + 1}/{args.epochs}], Training Loss: {avg_loss:.4f}")

        # val_loss = validate_model(model, val_loader, criterion)
        # scheduler.step(val_loss)

        for param_group in optimizer.param_groups:
            print("Current learning rate:", param_group['lr'])
        if args.rank == 0:
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_weights.pth'))
            #     print(f"Best model weights saved for epoch {epoch + 1}.")

            torch.save(model.state_dict(), os.path.join(output_dir, f'model_weights_epoch_{epoch + 1}.pth'))

if __name__ == "__main__":
    memory_thread = threading.Thread(target=print_memory_usage, daemon=True)
    memory_thread.start()
    timestamp = int(time.time())
    timestamp = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H:%M:%S')
    output_dir = f"train_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, f"train_{timestamp}.log")
    redirect_output_to_log(log_file)

    parser = ArgumentParser()
    parser.add_argument("--model_size", type=str, default="base", choices=["base", "large"], help="Model size.")
    parser.add_argument("--dataset", type=str, default="happy", choices=["fiq-dress", "fiq-shirt", "fiq-toptee", "circo", "dtin", "happy"], help="Dataset selection.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--rank", type=int, default=0, help="Rank of the process.")

    args = parser.parse_args()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = MagicLens(args.model_size).to(device)

    if torch.cuda.device_count() > 1:
        # model = nn.DataParallel(model)
        model = nn.DataParallel(model, device_ids=[2, 3])


    tokenizer = clip.simple_tokenizer.SimpleTokenizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.8)
    criterion = contrastive_loss

    if args.dataset.startswith("fiq"):
        subtask = args.dataset.split("-")[1]
        train_dataset, train_loader= build_fiq_dataset_for_train(dataset_name=args.dataset)
        val_dataset, val_loader = build_fiq_dataset_for_val(dataset_name=args.dataset)
    # elif args.dataset in ["circo"]:
        # train_dataset = build_circo_dataset_for_train(dataset_name=args.dataset, tokenizer=tokenizer)
        # val_dataset = build_circo_dataset(dataset_name=args.dataset, tokenizer=tokenizer)
    elif args.dataset in ["happy"]:
        train_dataset, train_loader = build_happy_dataset_for_train(dataset_name=args.dataset)
        val_dataset, val_loader = build_happy_dataset_for_val(dataset_name=args.dataset)
    else:
        raise NotImplementedError

    train_model(model, train_loader, optimizer, criterion, args)

    print("Training Done.")

