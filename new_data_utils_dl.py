import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from CLIP.clip import tokenize
from typing import Any, List, Union, Tuple
import glob

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

    # 如果 retrieved_iids 或 retrieved_scores 的长度不相同，可能需要做额外的处理，比如填充
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

def process_img(image_path: str, size: int) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((size, size), Image.BILINEAR)
        return np.array(img) / 255.0 

class HAPPYDataset(Dataset):
    def __init__(self, dataset_name):
        self.name = dataset_name
        self.k_range = [10, 50]

        self.queries = []
        for i in range(175):  
            file_name = "/home/zt/data/open-images/train/processed_nn1/response_results_batch_{}.json".format(i)
            if glob.glob(file_name): 
                with open(file_name) as f:
                    self.queries.extend(json.load(f))
        self.index_img_ids = json.load(open(f"/home/zt/data/open-images/train/processed_nn1/index.json"))
        self.index_image_folder = "/home/zt/data/open-images/train/data"

        self.query_data = []
        for query in self.queries:
            self.query_data.append({
                'qid': query['candidate'],
                'qtext':query['captions'],
                'target_iid': query['target']
            })

    def __len__(self):
        return len(self.query_data)

    def __getitem__(self, idx):
        query_info = self.query_data[idx]
        qid = query_info['qid']
        qtext = query_info['qtext']
        target_iid = query_info['target_iid']
        
        qimage_path = os.path.join(self.index_image_folder, qid + ".jpg")
        if not os.path.exists(qimage_path):
            print(f"Image not found: {qimage_path}")
            return None
        qimage = process_img(qimage_path, 224) 

        qtokens = tokenize(qtext) 
        query_example = QueryExample(qid=qid, qtokens=qtokens, qimage=qimage, target_iid=target_iid)

        index_img_path = os.path.join(self.index_image_folder, str(target_iid) + ".jpg")
        if not os.path.exists(index_img_path):
            print(f"Index image not found: {index_img_path}")
            return None
        index_image = process_img(index_img_path, 224) 

        null_tokens = tokenize("") 
        index_example = IndexExample(iid=target_iid, iimage=index_image, itokens=null_tokens)

        return {
            'query': query_example,
            'index': index_example
        }

    def evaluate_recall(self):
        ret_dict = {k: [] for k in self.k_range}  
        with open('retrieved_iids_output_happy.txt', 'w') as f: 
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

class HAPPYDatasetVAL(Dataset):
    def __init__(self, dataset_name):
        self.name = dataset_name
        self.k_range = [10, 50]

        self.queries = []
        for i in range(2):  
            file_name = "/home/zt/data/open-images/train/processed_nn3/response_results_batch_{}.json".format(i)
            if glob.glob(file_name): 
                with open(file_name) as f:
                    self.queries.extend(json.load(f))
        self.index_img_ids = json.load(open(f"/home/zt/data/open-images/train/processed_nn3/index.json"))
        self.index_image_folder = "/home/zt/data/open-images/train/data"

        self.query_data = []
        for query in self.queries:
            self.query_data.append({
                'qid': query['candidate'],
                'qtext':query['captions'],
                'target_iid': query['target']
            })

    def __len__(self):
        return len(self.query_data)

    def __getitem__(self, idx):
        query_info = self.query_data[idx]
        qid = query_info['qid']
        qtext = query_info['qtext']
        target_iid = query_info['target_iid']
        
        qimage_path = os.path.join(self.index_image_folder, qid + ".jpg")
        if not os.path.exists(qimage_path):
            print(f"Image not found: {qimage_path}")
            return None
        qimage = process_img(qimage_path, 224) 

        qtokens = tokenize(qtext) 
        query_example = QueryExample(qid=qid, qtokens=qtokens, qimage=qimage, target_iid=target_iid)

        index_img_path = os.path.join(self.index_image_folder, str(target_iid) + ".jpg")
        if not os.path.exists(index_img_path):
            print(f"Index image not found: {index_img_path}")
            return None
        index_image = process_img(index_img_path, 224) 

        null_tokens = tokenize("") 
        index_example = IndexExample(iid=target_iid, iimage=index_image, itokens=null_tokens)

        return {
            'query': query_example,
            'index': index_example
        }


def build_happy_dataset_for_train(dataset_name: str, batch_size: int = 100) -> Tuple[HAPPYDataset, DataLoader]:
    dataset = HAPPYDataset(dataset_name)
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)  

def build_happy_dataset_for_val(dataset_name: str, batch_size: int = 100) -> Tuple[HAPPYDatasetVAL, DataLoader]:
    dataset = HAPPYDatasetVAL(dataset_name)
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)  


class FIQDataset(Dataset):
    def __init__(self, dataset_name):
        self.name = dataset_name
        self.k_range = [10, 50]
        
        subtask = dataset_name.split("-")[1]
        self.queries = json.load(open(f"./data/fiq/captions/cap.{subtask}.train.json"))
        self.index_img_ids = json.load(open(f"./data/fiq/image_splits/split.{subtask}.train.json"))
        self.index_image_folder = "./data/fiq/images"

        self.query_data = []
        for query in self.queries:
            self.query_data.append({
                'qid': query['candidate'],
                'qtext':query['captions'],
                'target_iid': query['target']
            })

    def __len__(self):
        return len(self.query_data)

    def __getitem__(self, idx):
        query_info = self.query_data[idx]
        qid = query_info['qid']
        qtext = query_info['qtext']
        target_iid = query_info['target_iid']
        
        qimage_path = os.path.join(self.index_image_folder, qid + ".png")
        if not os.path.exists(qimage_path):
            print(f"Image not found: {qimage_path}")
            return None
        qimage = process_img(qimage_path, 224) 

        qtokens = tokenize(qtext) 
        query_example = QueryExample(qid=qid, qtokens=qtokens, qimage=qimage, target_iid=target_iid)

        index_img_path = os.path.join(self.index_image_folder, str(target_iid) + ".png")
        if not os.path.exists(index_img_path):
            print(f"Index image not found: {index_img_path}")
            return None
        index_image = process_img(index_img_path, 224) 

        null_tokens = tokenize("") 
        index_example = IndexExample(iid=target_iid, iimage=index_image, itokens=null_tokens)

        return {
            'query': query_example,
            'index': index_example
        }


    def evaluate_recall(self):
        ret_dict = {k: [] for k in self.k_range}  
        with open('retrieved_iids_output_fiq.txt', 'w') as f: 
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

class FIQDatasetVAL(Dataset):
    def __init__(self, dataset_name):
        self.name = dataset_name
        self.k_range = [10, 50]
        
        subtask = dataset_name.split("-")[1]
        self.queries = json.load(open(f"./data/fiq/captions/cap.{subtask}.val.json"))
        self.index_img_ids = json.load(open(f"./data/fiq/image_splits/split.{subtask}.val.json"))
        self.index_image_folder = "./data/fiq/images"

        self.query_data = []
        for query in self.queries:
            self.query_data.append({
                'qid': query['candidate'],
                'qtext':query['captions'],
                'target_iid': query['target']
            })

    def __len__(self):
        return len(self.query_data)

    def __getitem__(self, idx):
        query_info = self.query_data[idx]
        qid = query_info['qid']
        qtext = query_info['qtext']
        target_iid = query_info['target_iid']
        
        qimage_path = os.path.join(self.index_image_folder, qid + ".png")
        if not os.path.exists(qimage_path):
            print(f"Image not found: {qimage_path}")
            return None
        qimage = process_img(qimage_path, 224) 

        qtokens = tokenize(qtext) 
        query_example = QueryExample(qid=qid, qtokens=qtokens, qimage=qimage, target_iid=target_iid)

        index_img_path = os.path.join(self.index_image_folder, str(target_iid) + ".png")
        if not os.path.exists(index_img_path):
            print(f"Index image not found: {index_img_path}")
            return None
        index_image = process_img(index_img_path, 224) 

        null_tokens = tokenize("") 
        index_example = IndexExample(iid=target_iid, iimage=index_image, itokens=null_tokens)

        return {
            'query': query_example,
            'index': index_example
        }


def build_fiq_dataset_for_train(dataset_name: str, batch_size: int = 100) -> Tuple[FIQDataset, DataLoader]:
    train_dataset = FIQDataset(dataset_name)
    return train_dataset, DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)  

def build_fiq_dataset_for_val(dataset_name: str, batch_size: int = 100) -> Tuple[FIQDatasetVAL, DataLoader]:
    val_dataset = FIQDatasetVAL(dataset_name)
    return val_dataset, DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)  
