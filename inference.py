# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import pickle
import torch
import numpy as np

import torch.nn.functional as F
from argparse import ArgumentParser
from data_utils import build_circo_dataset, build_fiq_dataset
from model import MagicLens
from tqdm import tqdm
import clip
from clip import tokenize

def load_model(model_size: str) -> tuple:
    # init model
    model = MagicLens(model_size)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = ("cpu")
    model.to(device).float()
    model.eval()  # Set model to evaluation mode
    # load model

    # state_dict = torch.load('/home/lulu/lulu/magic/magiclens/me_magic_lens_clip_base_114.pkl', map_location='cpu', weights_only=False)
    # model.load_state_dict(state_dict)
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    print("model loaded")
    return model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="model_weights_epoch_114.pth",
        help="The path to model directory.",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="base",
        help="Model size, choices: base, large.",
        choices=["base", "large"],
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fiq-shirt",
        help="Dataset selection.",
        choices=["fiq-dress", "fiq-shirt", "fiq-toptee", "circo", "dtin"],
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory of predictions top 50.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=50, help="Batch size for inference."
    )
    args = parser.parse_args()

    # load model
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()
    model = load_model(args.model_size)


    # load data
    if args.dataset.startswith("fiq"):
        subtask = args.dataset.split("-")[1]
        eval_dataset = build_fiq_dataset(dataset_name=args.dataset, tokenizer=tokenizer)
    elif args.dataset in ["circo"]:
        eval_dataset = build_circo_dataset(dataset_name=args.dataset, tokenizer=tokenizer)
    else:
        raise NotImplementedError
    
    
    # inference index:
    index_embeddings = []
    print("Inference index...")
    num_index_batches = int((len(eval_dataset.index_examples) - 4500) / args.batch_size) + 1
    for i in tqdm(range(num_index_batches)):
        batch = eval_dataset.index_examples[i * args.batch_size: (i + 1) * args.batch_size]
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        iids = [i.iid for i in batch]
        iimages = torch.stack([torch.from_numpy(i.iimage) for i in batch], dim=0).to(device)
        itokens = torch.stack([torch.from_numpy(i.itokens) for i in batch], dim=0).to(device)
        iembeds = model({"ids": itokens, "image": iimages})["multimodal_embed_norm"]
        # print(f"Shape of iembeds for batch {i}: {iembeds.shape}") # Shape of iembeds for batch 35: torch.Size([50, 512])
        index_embeddings.append(iembeds) 
    index_embeddings = torch.cat(index_embeddings, dim=0) 
    # print(f"Shape of index_embeddings: {index_embeddings.shape}") # Shape of index_embeddings: torch.Size([1850, 512]) 1850 = number of batch * batch size
 

    
    print("Inference queries...")
    num_query_batches = int((len(eval_dataset.query_examples) - 1000)/ args.batch_size) + 1
    for i in tqdm(range(num_query_batches)):
        batch = eval_dataset.query_examples[i * args.batch_size: (i + 1) * args.batch_size]
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        qiids = [q.qid for q in batch]
        qimages = torch.stack([torch.from_numpy(q.qimage) for q in batch], dim=0).to(device)
        qtokens = torch.stack([q.qtokens for q in batch], dim=0).to(device)
        qembeds = model({"ids": qtokens, "image": qimages})["multimodal_embed_norm"] 
        # print("qembeds shape: ", qembeds.shape)  # qembeds shape:  torch.Size([50, 512])
        similarity_scores = torch.matmul(qembeds, index_embeddings.T)

        # get top 50 by similarity
        top_k_indices = torch.topk(similarity_scores, k=50, dim=1).indices
        top_k_iids = [[eval_dataset.index_examples[idx].iid for idx in top_k] for top_k in top_k_indices]

        # gather scores for the top_k
        top_k_scores = [similarity_scores[i, tk].tolist() for i, tk in enumerate(top_k_indices)]
        # with open('output_t.txt', 'w') as f:
        #     f.write(f"Top K Indices: {top_k_indices.tolist()}\n")
        #     f.write(f"Top K IIDs: {top_k_iids}\n")
        #     f.write(f"Top K Scores: {top_k_scores}\n")
        
        # update the query_example with the retrieved results
        for k, q_example in enumerate(batch):
            q_example.retrieved_iids = top_k_iids[k]
            q_example.retrieved_scores = top_k_scores[k]
            eval_dataset.query_examples[i + k] = q_example
        with open('output_q_shirt.txt', 'a') as f:
            for k, q_example in enumerate(batch):
                f.write(f"Query Example {k}:\n")
                f.write(f"Retrieved IIDs: {top_k_iids[k]}\n")
                f.write(f"Retrieved Scores: {top_k_scores[k]}\n")
                f.write("\n")  # 添加一个换行以便于可读性
    # Post-processing and evaluation:
    if args.dataset in ["fiq-dress", "fiq-shirt", "fiq-toptee"]:
        eval_dataset.evaluate_recall()
    elif args.dataset in ["circo"]:
        eval_dataset.write_to_file(os.path.join(args.output, args.dataset + "_" + args.model_size))
    else:
        raise NotImplementedError

    print("Inference Done.")
