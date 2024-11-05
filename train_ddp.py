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
from data_dl_zll import build_happy_dataset_for_train
from torch.utils.data import Dataset, DataLoader
import psutil
torch.cuda.empty_cache()

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
        
        time.sleep(3)

def contrastive_loss(query_embeddings, target_embeddings, temperature=0.07):
    similarities = F.cosine_similarity(query_embeddings.unsqueeze(1), target_embeddings.unsqueeze(0), dim=2)
    logits = similarities / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss

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

def prepare_batch(batch, device):

    qimages = torch.stack([qimage for qimage in batch['qimage']], dim=0).to(device)
    qtokens = torch.stack([qtokens for qtokens in batch['qtokens']], dim=0).to(device)

    timages, ttokens_list = [], []
    for q in batch['target_iid']:
        target_iid = q
        if isinstance(target_iid, list):
            target_iimages = [
                next((index_example.iimage for index_example in train_dataset.index_examples if index_example.iid == iid), None)
                for iid in target_iid
            ]
            timages.append(torch.cat([torch.tensor(img) for img in target_iimages if img is not None]))

            target_tokens = np.array(tokenize("")).astype(np.float32)
            ttokens_list.append(torch.tensor(target_tokens))
        else:
            timage = next((index_example.iimage for index_example in train_dataset.index_examples if index_example.iid == target_iid), None)

            if timage is not None:
                timages.append(torch.tensor(timage))

                token = np.array(tokenize("")).astype(np.float32)
                ttokens_list.append(torch.tensor(token))
            else:
                raise ValueError(f"Target image with ID {target_iid} not found.")

    assert len(timages) == len(ttokens_list), "Number of images and tokens must match."
    ttokens = torch.stack(ttokens_list).to(device)
    timages = torch.stack(timages).to(device)

    return qimages, qtokens, timages, ttokens

def train_model(model, train_loader, optimizer, criterion, args):
    model.train().float()

    for epoch in range(args.epochs):
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{args.epochs}")):
            optimizer.zero_grad()

            qimages, qtokens, timages, ttokens = prepare_batch(batch, device)

            qoutput = model({"ids": qtokens, "image": qimages})
            query_embeddings = qoutput["multimodal_embed_norm"]

            toutput = model({"ids": ttokens, "image": timages})
            target_embeddings = toutput["multimodal_embed_norm"]

            loss = criterion(query_embeddings, target_embeddings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch + 1}/{args.epochs}], Training Loss: {avg_loss:.4f}")

        if args.rank == 0:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MagicLens(args.model_size).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    tokenizer = clip.simple_tokenizer.SimpleTokenizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = contrastive_loss

    if args.dataset in ["happy"]:
        train_dataset, train_loader = build_happy_dataset_for_train(dataset_name=args.dataset, tokenizer=tokenizer)
    else:
        raise NotImplementedError

    train_model(model, train_loader, optimizer, criterion, args)

    print("Training Done.")

