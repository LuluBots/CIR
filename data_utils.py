import torch.nn as nn
import os
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
from data_utils import build_circo_dataset, build_circo_dataset_for_train, build_fiq_dataset, build_fiq_dataset_for_train, build_happy_dataset, build_happy_dataset_for_train
torch.cuda.empty_cache()


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

def prepare_batch(batch, device, dataset):
    qimages = torch.stack([torch.from_numpy(q.qimage) for q in batch], dim=0).to(device)
    qtokens = torch.stack([q.qtokens for q in batch], dim=0).to(device)

    timages, ttokens_list = [], []
    for target_iid in [q.target_iid for q in batch]:
        if isinstance(target_iid, list):
            target_iimages = [
                next((index_example.iimage for index_example in dataset.index_examples if index_example.iid == iid), None)
                for iid in target_iid
            ]
            """
            target_iimages = [] 
            for iid in target_iid:
                found_image = None  
                for index_example in dataset.index_examples:
                    if index_example.iid == iid:  
                        found_image = index_example.iimage  
                        break 
                target_iimages.append(found_image)  
            """
            timages.append(torch.cat([torch.tensor(img) for img in target_iimages if img is not None]))

            target_tokens = np.array(tokenize("")).astype(np.float32)
            ttokens_list.append(torch.tensor(target_tokens))
        else:
            timage = next((index_example.iimage for index_example in dataset.index_examples if index_example.iid == target_iid), None)
            """
            timage = None
            for index_example in dataset.index_examples:
                if index_example.iid == target_iid:
                    timage = index_example.iimage
                    break
            """
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

# def validate_model(model, val_dataset, criterion, args):
#     # model.to(device).float()
#     model.float()
#     model.eval()
#     total_loss = 0.0
#     num_batches = int(len(val_dataset.query_examples) / args.batch_size)

#     with torch.no_grad():
#         for batch_idx in tqdm(range(num_batches), desc="Validating"):
#             batch = val_dataset.query_examples[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size]
#             qimages, qtokens, timages, ttokens = prepare_batch(batch, device, val_dataset)

#             qoutput = model({"ids": qtokens, "image": qimages})
#             query_embeddings = qoutput["multimodal_embed_norm"]

#             toutput = model({"ids": ttokens, "image": timages})
#             target_embeddings = toutput["multimodal_embed_norm"]

#             loss = criterion(query_embeddings, target_embeddings)
#             total_loss += loss.item()

#     avg_loss = total_loss / num_batches
#     print(f"Validation Loss: {avg_loss:.4f}")
#     return avg_loss

# def train_model(model, train_dataset, val_dataset, optimizer, criterion, args):
def train_model(model, train_dataset, optimizer, criterion, args):

    model.train().float()
    # best_val_loss = float('inf')

    for epoch in range(args.epochs):
        total_loss = 0.0
        num_batches = len(train_dataset.query_examples) // args.batch_size

        for batch_idx in tqdm(range(num_batches), desc=f"Training Epoch {epoch + 1}/{args.epochs}"):
            optimizer.zero_grad()

            batch = train_dataset.query_examples[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size]
            qimages, qtokens, timages, ttokens = prepare_batch(batch, device, train_dataset)

            # 将数据移到 GPU
            qimages = qimages.to(device)
            qtokens = qtokens.to(device)
            timages = timages.to(device)
            ttokens = ttokens.to(device)

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

        # val_loss = validate_model(model, val_dataset, criterion, args)
        if args.rank == 0:
        #     if val_loss < best_val_loss:
        #         best_val_loss = val_loss
        #         torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_weights.pth'))
        #         print(f"Best model weights saved for epoch {epoch + 1}.")

            torch.save(model.state_dict(), os.path.join(output_dir, f'model_weights_epoch_{epoch + 1}.pth'))

if __name__ == "__main__":

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
    parser.add_argument("--rank", type=int, default=0, help="Rank of the process.")  # 添加 rank 参数

    args = parser.parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = MagicLens(args.model_size).to(device)
    # 初始化模型和数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MagicLens(args.model_size).to(device)

    # 使用 DataParallel 包装模型
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.8)
    criterion = contrastive_loss

    if args.dataset.startswith("fiq"):
        subtask = args.dataset.split("-")[1]
        train_dataset = build_fiq_dataset_for_train(dataset_name=args.dataset, tokenizer=tokenizer)
        # val_dataset = build_fiq_dataset(dataset_name=args.dataset, tokenizer=tokenizer)
    elif args.dataset in ["circo"]:
        train_dataset = build_circo_dataset_for_train(dataset_name=args.dataset, tokenizer=tokenizer)
        # val_dataset = build_circo_dataset(dataset_name=args.dataset, tokenizer=tokenizer)
    elif args.dataset in ["happy"]:
        train_dataset = build_happy_dataset_for_train(dataset_name=args.dataset, tokenizer=tokenizer)
        # val_dataset = build_happy_dataset(dataset_name=args.dataset, tokenizer=tokenizer)

    else:
        raise NotImplementedError
    
    # train_model(model, train_dataset, val_dataset, optimizer, criterion, args) 
    train_model(model, train_dataset, optimizer, criterion, args) 


    print("Training Done.")
