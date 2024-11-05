import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from CLIP.clip import tokenize
from tqdm import tqdm
from typing import Any, List, Union, Tuple


class QueryExample:
    def __init__(self, 
                 qid: str, 
                 qimage: np.ndarray, 
                 qtokens: np.ndarray, 
                 target_iid: Union[int, str, List[int], List[str], None], 
                 retrieved_iids: List[Union[int, str]] = None, 
                 retrieved_scores: List[float] = None):
        self.qid = qid
        self.qimage = qimage
        self.qtokens = qtokens
        self.target_iid = target_iid
        self.retrieved_iids = retrieved_iids if retrieved_iids else []
        self.retrieved_scores = retrieved_scores if retrieved_scores else []

class IndexExample:
    def __init__(self, 
                 iid: Union[int, str],
                 iimage: np.ndarray,
                 itokens: np.ndarray):
        self.iid = iid
        self.iimage = iimage
        self.itokens = itokens

# def custom_collate_fn(batch):
#     qid = [item.qid for item in batch]
#     qtokens = [item.qtokens for item in batch]
#     qimage = [item.qimage for item in batch]
#     target_iid = [item.target_iid for item in batch]
#     retrieved_iids = [item.retrieved_iids for item in batch]
#     retrieved_scores = [item.retrieved_scores for item in batch]

#     # 将所有数据整理成字典或张量
#     return {
#         'qid': qid,
#         'qtokens': qtokens,
#         'qimage': np.stack(qimage), 
#         'target_iid': target_iid,
#         'retrieved_iids': retrieved_iids,
#         'retrieved_scores': retrieved_scores,
#     }

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
    qtokens = torch.tensor(np.array(qtokens))  # 假设 tokens 是二维数组

    # 如果 retrieved_iids 或 retrieved_scores 的长度不相同，你可能需要做额外的处理，比如填充
    retrieved_iid = [torch.tensor(i) for i in retrieved_iid]  # 可以选择将其转换为张量
    retrieved_scores = [torch.tensor(s) for s in retrieved_scores]  # 同样转换为张量

    # 对 index 示例进行处理（这里我们只关心 image 和 tokens）
    iid = [i.iid for i in indexes]
    iimage = torch.stack([torch.tensor(i.iimage).float() for i in indexes])
    itokens = [i.itokens for i in indexes]
    itokens = torch.tensor(np.array(itokens))  # 假设它们是二维数组

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

# class FIQDataset(Dataset):
#     def __init__(self, dataset_name, tokenizer, split='train'):
#         self.name = dataset_name
#         self.index_examples = []
#         self.query_examples = []
#         self.k_range = [10, 50]

#         subtask = dataset_name.split("-")[1]
#         queries = json.load(open(f"./data/fiq/captions/cap.{subtask}.{split}.json"))
#         index_img_ids = json.load(open(f"./data/fiq/image_splits/split.{subtask}.{split}.json"))
#         index_image_folder = "./data/fiq/images"

#         null_tokens = tokenize("")  
#         null_tokens = np.array(null_tokens)

#         # Prepare index examples
#         for index_img_id in index_img_ids:
#             img_path = os.path.join(index_image_folder, index_img_id + ".png")
#             ima = self.process_img(img_path, 224)
#             index_example = IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)
#             self.index_examples.append(index_example)

#         # Prepare query examples
#         for query in queries:
#             qid = query['candidate']
#             qtext = " and ".join(query['captions'])
#             qimage_path = os.path.join(index_image_folder, query['candidate'] + ".png")
#             ima = self.process_img(qimage_path, 224)
#             qtokens = tokenize(qtext)
#             query_example = QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=query['target'])
#             self.query_examples.append(query_example)

#     def process_img(self, image_path: str, size: int) -> np.ndarray:
#         img = Image.open(image_path).convert("RGB")
#         img = img.resize((size, size), Image.BILINEAR)
#         return np.array(img) / 255.0 

#     def __len__(self):
#         return len(self.query_examples)

#     # def __getitem__(self, idx):
#     #     return self.query_examples[idx]

#     def __getitem__(self, idx):
#         query_example = self.query_examples[idx]
#         # 找到与 query_example 相关的 index_example
#         index_example = self.index_examples[0]  # 或者根据需要选择合适的 index_example
#         return {
#             'query': query_example,
#             'index': index_example
#         }

def process_img(image_path: str, size: int) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((size, size), Image.BILINEAR)
        return np.array(img) / 255.0 

class FIQDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, split='train'):
        self.name = dataset_name
        self.k_range = [10, 50]

        subtask = dataset_name.split("-")[1]
        self.queries = json.load(open(f"./data/fiq/captions/cap.{subtask}.{split}.json"))
        self.index_img_ids = json.load(open(f"./data/fiq/image_splits/split.{subtask}.{split}.json"))
        self.index_image_folder = "./data/fiq/images"

        # 存储图像ID列表以及查询的文件信息，而不在内存中加载所有实例
        self.query_data = []
        for query in self.queries:
            self.query_data.append({
                'qid': query['candidate'],
                'qtext': " and ".join(query['captions']),
                'target_iid': query['target']
            })

    def __len__(self):
        return len(self.query_data)

    def __getitem__(self, idx):
        # 从存储的 query_data 获取单个查询数据
        query_info = self.query_data[idx]
        qid = query_info['qid']
        qtext = query_info['qtext']
        target_iid = query_info['target_iid']
        
        # 动态加载 query 示例
        qimage_path = os.path.join(self.index_image_folder, qid + ".png")
        if not os.path.exists(qimage_path):
            print(f"Image not found: {qimage_path}")
            return None
        qimage = process_img(qimage_path, 224)  # 对图像进行预处理

        qtokens = tokenize(qtext)  # 对文本进行tokenize
        query_example = QueryExample(qid=qid, qtokens=qtokens, qimage=qimage, target_iid=target_iid)

        # 动态加载对应的 index 示例
        index_img_path = os.path.join(self.index_image_folder, str(target_iid) + ".png")
        if not os.path.exists(index_img_path):
            print(f"Index image not found: {index_img_path}")
            return None
        index_image = process_img(index_img_path, 224)  # 处理目标图像

        null_tokens = tokenize("")  # 可以调整为目标图像的token或空的token
        index_example = IndexExample(iid=target_iid, iimage=index_image, itokens=null_tokens)

        return {
            'query': query_example,
            'index': index_example
        }

# class FIQDataset(Dataset):
#     def __init__(self, dataset_name, tokenizer, split='train'):
#         self.name = dataset_name
#         self.query_examples = []
#         self.index_examples = []  # 用来存储索引图像
#         self.k_range = [10, 50]

#         subtask = dataset_name.split("-")[1]
#         queries = json.load(open(f"./data/fiq/captions/cap.{subtask}.{split}.json"))
#         index_img_ids = json.load(open(f"./data/fiq/image_splits/split.{subtask}.{split}.json"))
#         index_image_folder = "./data/fiq/images"

#         for query in queries:
#             qid = query['candidate']
#             qtext = " and ".join(query['captions'])
#             qimage_path = os.path.join(index_image_folder, query['candidate'] + ".png")
#             ima = process_img(qimage_path, 224)
#             qtokens = tokenize(qtext)
#             query_example = QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=query['target'])
#             self.query_examples.append(query_example)

#             # 将目标图像也添加到 index_examples
#             target_iid = query['target']
#             index_img_path = os.path.join(index_image_folder, str(target_iid) + ".png")
#             index_image = process_img(index_img_path, 224)
#             index_example = IndexExample(iid=target_iid, iimage=index_image, itokens=tokenize(""))
#             self.index_examples.append(index_example)

#         # Store image ids instead of loading them all
#         self.index_img_ids = index_img_ids
#         self.index_image_folder = index_image_folder

#     def __len__(self):
#         return len(self.query_examples)

#     def __getitem__(self, idx):
#         query_example = self.query_examples[idx]
#         target_iid = query_example.target_iid
        
#         # Find the corresponding index image (delay loading)
#         index_img_path = os.path.join(self.index_image_folder, str(target_iid) + ".png")
#         index_image = process_img(index_img_path, 224)

#         # Load tokens if needed (for example, we could store tokens on disk and load them here)
#         null_tokens = tokenize("")  # Placeholder, could be modified

#         index_example = IndexExample(iid=target_iid, iimage=index_image, itokens=null_tokens)

#         return {
#             'query': query_example,
#             'index': index_example
#         }

    def evaluate_recall(self):
        ret_dict = {k: [] for k in self.k_range}  
        with open('retrieved_iids_output_shirt.txt', 'w') as f: 
            for q_example in self.query_examples: 
                assert len(q_example.retrieved_iids) > 0, "retrieved_iids is empty" 
                f.write(f"Query ids: {q_example.qid}\n")
                f.write(f"Query tokens: {q_example.qtokens}\n")
                f.write(f"Retrieved IIDs: {q_example.retrieved_iids}\n")
        
                for k in self.k_range: 
                    recalled = False 
                    if isinstance(q_example.target_iid, list):  
                        for one_target_iid in q_example.target_iid:  
                            if one_target_iid in q_example.retrieved_iids[:k]:  
                                recalled = True  
                    elif isinstance(q_example.target_iid, int) or isinstance(q_example.target_iid, str): 
                        if q_example.target_iid in q_example.retrieved_iids[:k]:  
                            recalled = True  
                    else:
                        raise ValueError(f"target_iid is of type {type(q_example.target_iid)}") 

                    if recalled:  
                        ret_dict[k].append(1)  
                    else:
                        ret_dict[k].append(0) 

        total_ex = len(self.query_examples) 
        ret_dict = {k: (sum(v) / total_ex) * 100 for k, v in ret_dict.items()} 
        print("Recalls: ", ret_dict) 

        return ret_dict  

    def write_to_file(self, output_dir: str):
        if not os.path.exists(output_dir):  
            os.makedirs(output_dir)  

        dict_to_write = dict()
        for q_example in self.query_examples: 
            dict_to_write[q_example.qid] = q_example.retrieved_iids[:50]  
        output_file = os.path.join(output_dir, f"{self.name}_results.json")
        with open(output_file, "w") as f:  
            json.dump(dict_to_write, f, indent=4)  
        print("Results are written to file", output_file)  

def build_fiq_dataset_for_train(dataset_name: str, tokenizer: Any, batch_size: int = 100) -> Tuple[FIQDataset, DataLoader]:
    dataset = FIQDataset(dataset_name, tokenizer, split='train')
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)  # 
"""
collate_fn 是 PyTorch 中 DataLoader 的一个参数，用于自定义数据加载过程中如何将一个 batch 的数据合并在一起。默认情况下，DataLoader 会将数据样本堆叠成一个批量（batch），但在某些情况下，数据的形状或类型可能不一致，这时候就需要自定义 collate_fn。

"""

# 使用示例
# tokenizer = YourTokenizer()  # 需要用实际的 tokenizer 实例替换
# dataloader = build_fiq_dataset_for_train("your-dataset-name", tokenizer)
