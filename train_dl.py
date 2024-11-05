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
from data_utils_dl import build_fiq_dataset_for_train
from torch.utils.data import Dataset, DataLoader
import psutil
torch.cuda.empty_cache()

def print_memory_usage():
    """每隔10秒钟输出当前CPU和GPU的内存占用"""
    while True:
        # 获取CPU内存使用情况
        cpu_memory = psutil.virtual_memory().percent
        print(f"CPU memory usage: {cpu_memory}%")
        
        # 获取GPU内存使用情况
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)  # 转换为GB
            gpu_memory_cached = torch.cuda.memory_reserved() / (1024 ** 3)  # 转换为GB
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

# def prepare_batch(batch, device):
#     # print("Batch type:", type(batch))
#     # Batch type: <class 'dict'>
#     # print("Batch qid contents:", batch['qid'])
#     #  ['B006WJ7JK2', 'B005J92ZC8', 'B007WAEN86', 'B00A13GNUM', 'B00DSDMNCE', 'B008KFJTPY', 'B007SV9DBQ', 'B00ES83OGW', 'B008PHQEDW', 'B002SNALOM', 'B00E9045S0', 'B00C6KR7A8', 'B00DSGN0NW', 'B008BBCRB0', 'B006ATD8CM', 'B008QFSW8I', 'B00FMB2R62', 'B00CIAIANE', 'B008ATH876', 'B004J254Z6', 'B00B2EAOA4', 'B00EVQVGSY', 'B00CQ0A0FC', 'B00DIGEDL0', 'B008VPX4IQ', 'B003TW4N7M', 'B009PNEAUE', 'B00BTHS1EK', 'B00DBD76KK', 'B00CLF4KJY', 'B00C7PAYRA', 'B0036DDYZQ', 'B006WM44AM', 'B008QW7UFC', 'B003QCISDK', 'B00DJGCPD2', 'B008BPV244', 'B008I2VXU8', 'B002ZG7MQW', 'B00ARNR9A4', 'B008CPTBIC', 'B007WADT0O', 'B004LLIRX6', 'B008AHK4C4', 'B008V5OKTI', 'B00EIQK9WQ', 'B00BIY352M', 'B00ARFW3SU', 'B003ILC4I4', 'B008LRMYRQ', 'B00DW39KU8', 'B00DSQQQ7Y', 'B005X4PFE4', 'B00428MWGQ', 'B009E2E0RE', 'B00B9R6GOC', 'B002UNLUM2', 'B00CQSTQCW', 'B008PQIE8G', 'B007V0B8IU', 'B00G0G376M', 'B005MKC6QE', 'B004U9YUG2', 'B007SDGT2K', 'B00BQUDEZ6', 'B00CA915S0', 'B007XD5R4Q', 'B00DPEC2JU', 'B005WLGR40', 'B00AIXFL58', 'B004SG6YYI', 'B004071W38', 'B008BRFCDO', 'B004SG6YYI', 'B0085966OW', 'B00BUUFHAW', 'B00AJ36Y5I', 'B00A67G1OG', 'B009WJ0LYA', 'B005G16BBK', 'B00BX9I73Y', 'B00CGDKACC', 'B00ARAXBAE', 'B0063R8BIW', 'B006WQW6K8', 'B008BWODAM', 'B0098NURY4', 'B008OC8IF0', 'B00BEW8ZO6', 'B009017BRA', 'B00EYPSU4A', 'B00CG60LQE', 'B005JD4RRA', 'B00COV7K5Q', 'B00BX9I73Y', 'B005JR1ISW', 'B00CTA12Q0', 'B00C3ZAXR0', 'B002HRFARW', 'B008O7JPA2']
#     # print("Batch qtokens contents:", batch['qtokens'])
#     """
#     [tensor([[49406,   533,  6148,  1449,   537,   533,  5598,  1746, 49407,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0]], dtype=torch.int32), tensor([[49406,   791,   681, 19344,   537,   533,   320,  3005,  3360,  2595,
#            269,   537,   533, 14102,  3360,   530,  3140,  2193, 19691,   269,
#          49407,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#              0,     0,     0,     0,     0,     0,     0]], dtype=torch.int32), 
#         """
#     # print("Batch qimage contents:", batch['qimage'])
#     """
#     [[[[1. 1. 1.]
#    [1. 1. 1.]
#    [1. 1. 1.]
#    ...
#    [1. 1. 1.]
#    [1. 1. 1.]
#    [1. 1. 1.]]

#   [[1. 1. 1.]
#    [1. 1. 1.]
#    [1. 1. 1.]
#    ...
#    [1. 1. 1.]
#    [1. 1. 1.]
#    [1. 1. 1.]]
#     """
#     # print("Types in batch:", [type(q) for q in batch])
#     # Types in batch: [<class 'str'>, <class 'str'>, <class 'str'>, <class 'str'>, <class 'str'>, <class 'str'>]

#     # qimages = torch.stack([torch.from_numpy(q.qimage) for q in batch], dim=0).to(device)
#     # qtokens = torch.stack([q.qtokens for q in batch], dim=0).to(device)

#     qimages = torch.stack([qimage for qimage in batch['qimage']], dim=0).to(device)
#     qtokens = torch.stack([qtokens for qtokens in batch['qtokens']], dim=0).to(device)

#     timages, ttokens_list = [], []
#     for q in batch['target_iid']:
#         target_iid = q
#         if isinstance(target_iid, list):
#             target_iimages = [
#                 next((index_example.iimage for index_example in train_dataset.index_examples if index_example.iid == iid), None)
#                 for iid in target_iid
#             ]
#             timages.append(torch.cat([torch.tensor(img) for img in target_iimages if img is not None]))

#             target_tokens = np.array(tokenize("")).astype(np.float32)
#             ttokens_list.append(torch.tensor(target_tokens))
#         else:
#             timage = next((index_example.iimage for index_example in train_dataset.index_examples if index_example.iid == target_iid), None)

#             if timage is not None:
#                 timages.append(torch.tensor(timage))

#                 token = np.array(tokenize("")).astype(np.float32)
#                 ttokens_list.append(torch.tensor(token))
#             else:
#                 raise ValueError(f"Target image with ID {target_iid} not found.")

#     assert len(timages) == len(ttokens_list), "Number of images and tokens must match."
#     ttokens = torch.stack(ttokens_list).to(device)
#     timages = torch.stack(timages).to(device)

#     return qimages, qtokens, timages, ttokens

from PIL import Image
def process_img(image_path: str, size: int) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((size, size), Image.BILINEAR)
        return np.array(img) / 255.0 

def prepare_batch(batch, device, train_dataset):

    # 处理查询图像和文本
    qimages = torch.stack([qimage for qimage in batch['qimage']], dim=0).to(device)
    qtokens = torch.stack([qtokens for qtokens in batch['qtokens']], dim=0).to(device)

    timages, ttokens_list = [], []
    
    # 遍历每个目标图像ID
    for q in batch['target_iid']:
        target_iid = q
        
        if isinstance(target_iid, list):
            # 如果是多个目标图像ID
            target_iimages = [
                # 动态加载目标图像
                process_img(os.path.join(train_dataset.index_image_folder, f"{iid}.png"), 224)
                for iid in target_iid
            ]
            # 合并所有图像
            timages.append(torch.cat([torch.tensor(img) for img in target_iimages if img is not None]))

            # 使用空的token或者根据需要修改
            target_tokens = np.array(tokenize("")).astype(np.float32)
            ttokens_list.append(torch.tensor(target_tokens))
        else:
            # 如果是单个目标图像ID
            index_img_path = os.path.join(train_dataset.index_image_folder, f"{target_iid}.png")
            
            # 动态加载目标图像
            timage = process_img(index_img_path, 224)  # 调整为动态加载图像的方式
            
            if timage is not None:
                timages.append(torch.tensor(timage))

                # 使用空的token或者根据需要修改
                token = np.array(tokenize("")).astype(np.float32)
                ttokens_list.append(torch.tensor(token))
            else:
                raise ValueError(f"Target image with ID {target_iid} not found.")
    
    # 检查图像和token数量是否匹配
    assert len(timages) == len(ttokens_list), "Number of images and tokens must match."
    
    # 合并所有目标图像和tokens
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

            qimages, qtokens, timages, ttokens = prepare_batch(batch, device, train_dataset)

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
    parser.add_argument("--dataset", type=str, default="fiq-dress", choices=["fiq-dress", "fiq-shirt", "fiq-toptee", "circo", "dtin", "happy"], help="Dataset selection.")
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

    if args.dataset.startswith("fiq"):
        subtask = args.dataset.split("-")[1]
        train_dataset, train_loader = build_fiq_dataset_for_train(dataset_name=args.dataset, tokenizer=tokenizer)
    # elif args.dataset in ["circo"]:
    #     train_dataset = build_circo_dataset_for_train(dataset_name=args.dataset, tokenizer=tokenizer)
    # elif args.dataset in ["happy"]:
    #     train_dataset = build_happy_dataset_for_train(dataset_name=args.dataset, tokenizer=tokenizer)
    else:
        raise NotImplementedError
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # 启动内存监控的线程
    # 启动内存监控的线程（在训练前就开始）

    train_model(model, train_loader, optimizer, criterion, args)

    print("Training Done.")

