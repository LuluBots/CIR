import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import glob
from typing import Any, List, Union
from tqdm import tqdm
import gc
import torch
import numpy as np
from PIL import Image
from CLIP.clip import tokenize
from CLIP.clip.simple_tokenizer import SimpleTokenizer

@dataclass
class QueryExample:
    qid: str
    qtokens: np.ndarray
    qimage: np.ndarray
    target_iid: Union[int, str, List[int], List[str], None] # can be int or 
    retrieved_iids: List[Union[int, str]] # ranked by score, can be str (cirr) or int (circo)
    retrieved_scores: List[float] # ranked by order

@dataclass
class IndexExample:
    iid: Union[int, str]
    iimage: np.ndarray
    itokens: np.ndarray

@dataclass
class Dataset:
    name: str  # 数据集名称
    query_examples: List[QueryExample] = field(default_factory=list)  # 查询示例列表
    k_range: List[int] = field(default_factory=lambda: [10, 50])  # k的范围，默认是[10, 50]
    index_examples: List[IndexExample] = field(default_factory=list)  # 索引示例列表

    def evaluate_recall(self):
        ret_dict = {k: [] for k in self.k_range}  # 初始化返回字典，记录每个k值的召回情况
        with open('retrieved_iids_output_shirt.txt', 'w') as f: 
            for q_example in self.query_examples[:-1000]:  # 遍历每个查询示例
                assert len(q_example.retrieved_iids) > 0, "retrieved_iids is empty"  # 可选：确保检索的iid不为空
                f.write(f"Query ids: {q_example.qid}\n")
                f.write(f"Query tokens: {q_example.qtokens}\n")
                f.write(f"Retrieved IIDs: {q_example.retrieved_iids}\n")
        
                for k in self.k_range:  # 对于每个k值
                    recalled = False  # 初始化召回标志为False
                    if isinstance(q_example.target_iid, list):  # 如果目标iid是列表
                        for one_target_iid in q_example.target_iid:  # 遍历每个目标iid
                            if one_target_iid in q_example.retrieved_iids[:k]:  # 检查目标iid是否在检索结果中
                                recalled = True  # 如果找到，设置召回标志为True
                    elif isinstance(q_example.target_iid, int) or isinstance(q_example.target_iid, str):  # 如果目标iid是int或str
                        if q_example.target_iid in q_example.retrieved_iids[:k]:  # 检查目标iid是否在检索结果中
                            recalled = True  # 如果找到，设置召回标志为True
                    else:
                        raise ValueError(f"target_iid is of type {type(q_example.target_iid)}")  # 抛出异常，类型不匹配

                    if recalled:  # 如果召回成功
                        ret_dict[k].append(1)  # 记录1
                    else:
                        ret_dict[k].append(0)  # 记录0

        total_ex = len(self.query_examples) - 1000 # 查询示例的总数
        ret_dict = {k: (sum(v) / total_ex) * 100 for k, v in ret_dict.items()}  # 计算召回率并转为百分比
        print("Recalls: ", ret_dict)  # 打印召回率

        return ret_dict  # 返回召回率字典

    def write_to_file(self, output_dir: str):
        if not os.path.exists(output_dir):  # 检查输出目录是否存在
            os.makedirs(output_dir)  # 如果不存在，则创建目录

        dict_to_write = dict()  # 初始化待写入的字典
        for q_example in self.query_examples:  # 遍历每个查询示例
            dict_to_write[q_example.qid] = q_example.retrieved_iids[:50]  # 记录查询ID及其检索结果（前50个）
        output_file = os.path.join(output_dir, f"{self.name}_results.json")  # 构建输出文件路径
        with open(output_file, "w") as f:  # 打开文件进行写入
            json.dump(dict_to_write, f, indent=4)  # 将字典写入JSON文件
        print("Results are written to file", output_file)  # 打印写入完成信息

def process_img(image_path: str, size: int) -> np.ndarray:
    """Process a single image to 224x224 and normalize."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    ima = np.array(img) / 255.0  # Normalize to [0, 1]
    return ima  # [H, W, C]


def build_fiq_dataset(dataset_name: str, tokenizer: Any) -> Dataset:
    eval_dataset = Dataset(dataset_name)
    subtask = dataset_name.split("-")[1]
    queries = json.load(open(f"./data/fiq/captions/cap.{subtask}.val.json"))
    index_img_ids = json.load(open(f"./data/fiq/image_splits/split.{subtask}.val.json"))
    index_image_folder = "./data/fiq/images"

    null_tokens = tokenize("")  # used for index example
    null_tokens = np.array(null_tokens)

    def process_index_example(index_img_id):
        img_path = os.path.join(index_image_folder, index_img_id + ".png")
        ima = process_img(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)

    def process_query_example(query):
        qid = query['candidate']
        qtext = " and ".join(query['captions'])
        qimage_path = os.path.join(index_image_folder, query['candidate'] + ".png")
        ima = process_img(qimage_path, 224)
        qtokens = tokenize(qtext)
        return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=query['target'], retrieved_iids=[], retrieved_scores=[])

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in index_img_ids}

        with tqdm(total=len(index_img_ids), desc="Index examples") as progress:
            for future in as_completed(index_example_futures):
                index_example = future.result()
                eval_dataset.index_examples.append(index_example)
                progress.update(1)

        

        print("Preparing query examples...")
        query_futures = {executor.submit(process_query_example, query): query for query in queries}

        with tqdm(total=len(queries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                eval_dataset.query_examples.append(q_example)
                progress.update(1)
            
        


    return eval_dataset

def build_fiq_dataset_for_train(dataset_name: str, tokenizer: Any) -> Dataset:
    train_dataset = Dataset(dataset_name)
    subtask = dataset_name.split("-")[1]
    queries = json.load(open(f"./data/fiq/captions/cap.{subtask}.train.json"))
    index_img_ids = json.load(open(f"./data/fiq/image_splits/split.{subtask}.train.json"))
    index_image_folder = "./data/fiq/images"

    null_tokens = tokenize("")  # used for index example
    null_tokens = np.array(null_tokens)

    def process_index_example(index_img_id):
        img_path = os.path.join(index_image_folder, index_img_id + ".png")
        ima = process_img(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)

    def process_query_example(query):
        qid = query['candidate']
        qtext = " and ".join(query['captions'])
        qimage_path = os.path.join(index_image_folder, query['candidate'] + ".png")
        ima = process_img(qimage_path, 224)
        qtokens = tokenize(qtext)
        return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=query['target'], retrieved_iids=[], retrieved_scores=[])

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in index_img_ids}

        with tqdm(total=len(index_img_ids), desc="Index examples") as progress:
            for future in as_completed(index_example_futures):
                index_example = future.result()
                train_dataset.index_examples.append(index_example)
                progress.update(1)

        

        print("Preparing query examples...")
        query_futures = {executor.submit(process_query_example, query): query for query in queries}

        with tqdm(total=len(queries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                train_dataset.query_examples.append(q_example)
                progress.update(1)
        
        

    return train_dataset

def build_circo_dataset(dataset_name: str, tokenizer: Any) -> Dataset:
    eval_dataset = Dataset(dataset_name)
    queries = json.load(open("./data/circo/annotations/val.json"))
    coco_info = json.load(open("./data/circo/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json"))
    index_img_ids = [img_info['id'] for img_info in coco_info['images']]
    index_image_folder = "./data/circo/COCO2017_unlabeled/unlabeled2017"

    def image_id2name(image_id):
        return str(image_id).zfill(12) + '.jpg'

    null_tokens = tokenize("")  # used for index example
    null_tokens = np.array(null_tokens)

    def process_index_example(index_img_id):
        img_path = os.path.join(index_image_folder, image_id2name(index_img_id))
        ima = process_img(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)

    def process_query_example(query):
        qid = query['id']
        qtext = f"find {query['shared_concept']} but {query['relative_caption']}"
        qimage_path = os.path.join(index_image_folder, image_id2name(query['reference_img_id']))
        ima = process_img(qimage_path, 224)
        qtokens = np.array(tokenize(qtext))
        return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=0, retrieved_iids=[], retrieved_scores=[])

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in index_img_ids}

        with tqdm(total=len(index_img_ids), desc="Index examples") as progress:
            for future in as_completed(index_example_futures):
                index_example = future.result()
                eval_dataset.index_examples.append(index_example)
                progress.update(1)

        

        print("Preparing query examples...")
        query_futures = {executor.submit(process_query_example, query): query for query in queries}

        with tqdm(total=len(queries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                eval_dataset.query_examples.append(q_example)
                progress.update(1)
        

    return eval_dataset

def build_circo_dataset_for_train(dataset_name: str, tokenizer: Any) -> Dataset:
    train_dataset = Dataset(dataset_name)
    queries = json.load(open("./data/circo/annotations/train.json"))
    coco_info = json.load(open("./data/circo/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json"))
    index_img_ids = [img_info['id'] for img_info in coco_info['images']]
    index_image_folder = "./data/circo/COCO2017_unlabeled/unlabeled2017"

    def image_id2name(image_id):
        return str(image_id).zfill(12) + '.jpg'

    null_tokens = tokenize("")  # used for index example
    null_tokens = np.array(null_tokens)

    def process_index_example(index_img_id):
        img_path = os.path.join(index_image_folder, image_id2name(index_img_id))
        ima = process_img(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)

    def process_query_example(query):
        qid = query['id']
        qtext = f"find {query['shared_concept']} but {query['relative_caption']}"
        qimage_path = os.path.join(index_image_folder, image_id2name(query['reference_img_id']))
        ima = process_img(qimage_path, 224)
        qtokens = np.array(tokenize(qtext))
        return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=0, retrieved_iids=[], retrieved_scores=[])

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in index_img_ids}

        with tqdm(total=len(index_img_ids), desc="Index examples") as progress:
            for future in as_completed(index_example_futures):
                index_example = future.result()
                train_dataset.index_examples.append(index_example)
                progress.update(1)

        

        print("Preparing query examples...")
        query_futures = {executor.submit(process_query_example, query): query for query in queries}

        with tqdm(total=len(queries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                train_dataset.query_examples.append(q_example)
                progress.update(1)
        

    return train_dataset

# def build_happy_dataset(dataset_name: str, tokenizer: Any) -> Dataset:
#     eval_dataset = Dataset(dataset_name)

#     queries = []
#     for file_name in glob.glob("/home/zt/data/open-images/train/processed_nn3/*.json"):
#         with open(file_name) as f:
#             queries.extend(json.load(f))
#     index_img_ids = json.load(open(f"/home/zt/data/open-images/train/metadata/image_id.json"))
#     index_image_folder = "/home/zt/data/open-images/train/data"

#     null_tokens = tokenize("")  # used for index example
#     null_tokens = np.array(null_tokens)

#     def process_index_example(index_img_id):
#         img_path = os.path.join(index_image_folder, index_img_id + ".jpg")
#         ima = process_img(img_path, 224)
#         return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)

#     def process_query_example(query):
#         qid = query['candidate']
#         qtext = " and ".join(query['captions'])
#         qimage_path = os.path.join(index_image_folder, query['candidate'] + ".jpg")
#         ima = process_img(qimage_path, 224)
#         qtokens = tokenize(qtext)
#         return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=query['target'], retrieved_iids=[], retrieved_scores=[])

#     with ThreadPoolExecutor() as executor:
#         print("Preparing index examples...")
#         index_example_futures = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in index_img_ids}

#         with tqdm(total=len(index_img_ids), desc="Index examples") as progress:
#             for future in as_completed(index_example_futures):
#                 index_example = future.result()
#                 eval_dataset.index_examples.append(index_example)
#                 progress.update(1)

#         

#         print("Preparing query examples...")
#         query_futures = {executor.submit(process_query_example, query): query for query in queries}

#         with tqdm(total=len(queries), desc="Query examples") as progress:
#             for future in as_completed(query_futures):
#                 q_example = future.result()
#                 eval_dataset.query_examples.append(q_example)
#                 progress.update(1)
        
#         

#     return eval_dataset


def write_index_example_to_file(index_example, file_path):
    """将 IndexExample 写入文件"""
    with open(file_path, 'a') as f:
        json.dump(index_example.__dict__, f)
        f.write('\n')  # 每个示例一行

def write_query_example_to_file(query_example, file_path):
    """将 QueryExample 写入文件"""
    with open(file_path, 'a') as f:
        json.dump(query_example.__dict__, f)
        f.write('\n')  # 每个示例一行

# def build_happy_dataset(dataset_name: str, tokenizer: Any, batch_size: int = 100000) -> Dataset:
def build_happy_dataset(dataset_name: str, tokenizer: Any, batch_size: int = 100000, index_output_file: str = 'eval_index_examples.json', query_output_file: str = 'eval_query_examples.json') -> Dataset:
    eval_dataset = Dataset(dataset_name)

    queries = []
    for file_name in glob.glob("/home/zt/data/open-images/train/processed_nn3/*.json"):
        with open(file_name) as f:
            queries.extend(json.load(f))
    index_img_ids = json.load(open(f"/home/zt/data/open-images/train/metadata/image_id.json"))
    index_image_folder = "/home/zt/data/open-images/train/data"

    null_tokens = tokenize("")  # used for index example
    null_tokens = np.array(null_tokens)

    def process_index_example(index_img_id):
        img_path = os.path.join(index_image_folder, index_img_id + ".jpg")
        ima = process_img(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)

    def process_query_example(query):
        qid = query['candidate']
        qtext = " and ".join(query['captions'])
        qimage_path = os.path.join(index_image_folder, query['candidate'] + ".jpg")
        ima = process_img(qimage_path, 224)
        qtokens = tokenize(qtext)
        return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=query['target'], retrieved_iids=[], retrieved_scores=[])

    def batch_generator(index_img_ids, batch_size):
        for i in range(0, len(index_img_ids), batch_size):
            yield index_img_ids[i:i + batch_size]

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = []
        for img_id_batch in batch_generator(index_img_ids, batch_size):
            future_batch = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in img_id_batch}
            # index_example_futures.extend(future_batch.items())

            # with tqdm(total=len(img_id_batch), desc="Index examples") as progress:
            #     for future in as_completed(future_batch):
            #         index_example = future.result()
            #         eval_dataset.index_examples.append(index_example)
            #         progress.update(1)
            with tqdm(total=len(img_id_batch), desc="Index examples") as progress:
                for future in as_completed(future_batch):
                    index_example = future.result()
                    # 写入文件以减少内存占用
                    write_index_example_to_file(index_example, index_output_file)

                    # 每处理一定数量后进行内存释放
                    if len(eval_dataset.index_examples) % 100000 == 0:
                        del index_example
                        gc.collect()
                    
                    progress.update(1)
        

        print("Preparing query examples...")
        query_futures = {executor.submit(process_query_example, query): query for query in queries}

        with tqdm(total=len(queries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                # eval_dataset.query_examples.append(q_example)
                # progress.update(1)
                # 写入文件以减少内存占用
                write_query_example_to_file(q_example, query_output_file)
                
                # 每处理一定数量后进行内存释放
                if len(eval_dataset.query_examples) % 100000 == 0:
                    del q_example
                    gc.collect()

                progress.update(1)

    return eval_dataset

# def build_happy_dataset_for_train(dataset_name: str, tokenizer: Any) -> Dataset:
#     train_dataset = Dataset(dataset_name)

#     # queries = []
#     # for file_name in glob.glob("/home/zt/data/open-images/train/processed_nn1/*.json") + glob.glob("/home/zt/data/open-images/train/processed_nn2/*.json"):
#     #     with open(file_name) as f:
#     #         queries.extend(json.load(f))
#     # index_img_ids = json.load(open(f"/home/zt/data/open-images/train/metadata/image_id.json"))
#     # index_image_folder = "/home/zt/data/open-images/train/data"

#     queries = []
#     file_path_pattern = "/home/zt/data/open-images/train/processed_nn1/response_results_batch_[0-9].json"       
#     files = glob.glob(file_path_pattern)
#     for file_name in files:
#         with open(file_name) as f:
#             queries.extend(json.load(f))
#     with open(f"/home/zt/data/open-images/train/metadata/image_id.json") as f:
#         index_img_ids = json.load(f)[:200000]
#     index_image_folder = "/home/zt/data/open-images/train/data" 
    
#     null_tokens = tokenize("")  # used for index example
#     null_tokens = np.array(null_tokens)

#     def process_index_example(index_img_id):
#         img_path = os.path.join(index_image_folder, index_img_id + ".jpg")
#         ima = process_img(img_path, 224)
#         return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)

#     def process_query_example(query):
#         qid = query['candidate']
#         qtext = " and ".join(query['captions'])
#         qimage_path = os.path.join(index_image_folder, query['candidate'] + ".jpg")
#         ima = process_img(qimage_path, 224)
#         qtokens = tokenize(qtext)
#         return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=query['target'], retrieved_iids=[], retrieved_scores=[])

#     with ThreadPoolExecutor() as executor:
#         print("Preparing index examples...")
#         index_example_futures = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in index_img_ids}

#         with tqdm(total=len(index_img_ids), desc="Index examples") as progress:
#             for future in as_completed(index_example_futures):
#                 index_example = future.result()
#                 train_dataset.index_examples.append(index_example)
#                 progress.update(1)

#         

#         print("Preparing query examples...")
#         query_futures = {executor.submit(process_query_example, query): query for query in queries}

#         with tqdm(total=len(queries), desc="Query examples") as progress:
#             for future in as_completed(query_futures):
#                 q_example = future.result()
#                 train_dataset.query_examples.append(q_example)
#                 progress.update(1)
            
#         


#     return train_dataset

# def build_happy_dataset_for_train(dataset_name: str, tokenizer: Any, batch_size: int = 100000) -> Dataset:
def build_happy_dataset_for_train(dataset_name: str, tokenizer: Any, batch_size: int = 100000, index_output_file: str = 'train_index_examples.json', query_output_file: str = 'train_query_examples.json') -> Dataset:

    train_dataset = Dataset(dataset_name)

    queries = []
    for file_name in glob.glob("/home/zt/data/open-images/train/processed_nn1/*.json") + glob.glob("/home/zt/data/open-images/train/processed_nn2/*.json"):
        with open(file_name) as f:
            queries.extend(json.load(f))
    index_img_ids = json.load(open(f"/home/zt/data/open-images/train/metadata/image_id.json"))
    index_image_folder = "/home/zt/data/open-images/train/data"

    null_tokens = tokenize("")  # used for index example
    null_tokens = np.array(null_tokens)

    def process_index_example(index_img_id):
        img_path = os.path.join(index_image_folder, index_img_id + ".jpg")
        ima = process_img(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)

    def process_query_example(query):
        qid = query['candidate']
        qtext = " and ".join(query['captions'])
        qimage_path = os.path.join(index_image_folder, query['candidate'] + ".jpg")
        ima = process_img(qimage_path, 224)
        qtokens = tokenize(qtext)
        return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=query['target'], retrieved_iids=[], retrieved_scores=[])

    def batch_generator(index_img_ids, batch_size):
        for i in range(0, len(index_img_ids), batch_size):
            yield index_img_ids[i:i + batch_size]

    # with ThreadPoolExecutor() as executor:
    #     print("Preparing index examples...")
    #     index_example_futures = []
    #     for img_id_batch in batch_generator(index_img_ids, batch_size):
    #         future_batch = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in img_id_batch}
    #         index_example_futures.extend(future_batch.items())

    #         with tqdm(total=len(img_id_batch), desc="Index examples") as progress:
    #             for future in as_completed(future_batch):
    #                 index_example = future.result()
    #                 train_dataset.index_examples.append(index_example)
    #                 progress.update(1)

        

    #     print("Preparing query examples...")
    #     query_futures = {executor.submit(process_query_example, query): query for query in queries}

    #     with tqdm(total=len(queries), desc="Query examples") as progress:
    #         for future in as_completed(query_futures):
    #             q_example = future.result()
    #             train_dataset.query_examples.append(q_example)
    #             progress.update(1)

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = []
        for img_id_batch in batch_generator(index_img_ids, batch_size):
            future_batch = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in img_id_batch}
            # index_example_futures.extend(future_batch.items())

            # with tqdm(total=len(img_id_batch), desc="Index examples") as progress:
            #     for future in as_completed(future_batch):
            #         index_example = future.result()
            #         eval_dataset.index_examples.append(index_example)
            #         progress.update(1)
            with tqdm(total=len(img_id_batch), desc="Index examples") as progress:
                for future in as_completed(future_batch):
                    index_example = future.result()
                    # 写入文件以减少内存占用
                    write_index_example_to_file(index_example, index_output_file)

                    # 每处理一定数量后进行内存释放
                    if len(train_dataset.index_examples) % 100000 == 0:
                        del index_example
                        gc.collect()
                    
                    progress.update(1)
        

        print("Preparing query examples...")
        query_futures = {executor.submit(process_query_example, query): query for query in queries}

        with tqdm(total=len(queries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                # eval_dataset.query_examples.append(q_example)
                # progress.update(1)
                # 写入文件以减少内存占用
                write_query_example_to_file(q_example, query_output_file)
                
                # 每处理一定数量后进行内存释放
                if len(train_dataset.query_examples) % 100000 == 0:
                    del q_example
                    gc.collect()

                progress.update(1)    
        

    return train_dataset
