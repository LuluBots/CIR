from tqdm import tqdm
import torch
import os
import json
from argparse import ArgumentParser
from data_utils import build_circo_dataset, build_fiq_dataset, build_happy_dataset
from model import MagicLens
import CLIP.clip as clip

def load_model(model_size: str) -> torch.nn.Module:
    model = MagicLens(model_size)
    model.eval()
    
    # 加载模型权重
    state_dict = torch.load(args.model_path, weights_only=True)

    # 移除前缀 'module.'
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')  # 去掉 'module.' 前缀
        new_state_dict[new_key] = v

    # 加载新的 state_dict
    model.load_state_dict(new_state_dict)
    
    print("Model loaded")
    return model

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/zt/ll/magiclens/model_weights_epoch_114.pth",
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
        choices=["fiq-dress", "fiq-shirt", "fiq-toptee", "circo", "dtin","happy"],
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

    # Load model
    tokenizer = clip.simple_tokenizer.SimpleTokenizer()
    model = load_model(args.model_size)

    # Use multiple GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Load data
    if args.dataset.startswith("fiq"):
        eval_dataset = build_fiq_dataset(dataset_name=args.dataset, tokenizer=tokenizer)
    elif args.dataset in ["circo"]:
        eval_dataset = build_circo_dataset(dataset_name=args.dataset, tokenizer=tokenizer)
    elif args.dataset in ["happy"]:
        val_dataset = build_happy_dataset(dataset_name=args.dataset, tokenizer=tokenizer)
    else:
        raise NotImplementedError
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).float()
    results = {}

    print("Inference queries...")
    num_query_batches = int(len(eval_dataset.query_examples) / args.batch_size) + 1
    top_k = 50  # number of top similar items to retrieve
    
    for i in tqdm(range(num_query_batches)):
        query_batch = eval_dataset.query_examples[i * args.batch_size: (i + 1) * args.batch_size]
        qimages = torch.stack([torch.from_numpy(q.qimage) for q in query_batch], dim=0).to(device)
        qtokens = torch.stack([q.qtokens for q in query_batch], dim=0).to(device)
        qembeds = model({"ids": qtokens, "image": qimages})["multimodal_embed_norm"]

        batch_top_k_iids = []
        batch_top_k_scores = []
        num_index_batches = int(len(eval_dataset.index_examples) / args.batch_size)

        for j in tqdm(range(num_index_batches)):
            index_batch = eval_dataset.index_examples[j * args.batch_size: (j + 1) * args.batch_size]
            iimages = torch.stack([torch.from_numpy(i.iimage) for i in index_batch], dim=0).to(device)
            itokens = torch.stack([torch.from_numpy(i.itokens) for i in index_batch], dim=0).to(device)
            iembeds = model({"ids": itokens, "image": iimages})["multimodal_embed_norm"]

            similarity_scores = torch.nn.functional.cosine_similarity(qembeds.unsqueeze(1), iembeds.unsqueeze(0), dim=2)

            for q_idx in range(similarity_scores.size(0)):
                top_k_indices_in_batch = torch.topk(similarity_scores[q_idx], k=1).indices
                top_k_scores_in_batch = similarity_scores[q_idx][top_k_indices_in_batch].tolist()
                top_k_iids_in_batch = [index_batch[idx].iid for idx in top_k_indices_in_batch]

                if len(batch_top_k_iids) <= q_idx:
                    batch_top_k_iids.append(top_k_iids_in_batch)
                    batch_top_k_scores.append(top_k_scores_in_batch)
                else:
                    combined_iids = batch_top_k_iids[q_idx] + top_k_iids_in_batch
                    combined_scores = batch_top_k_scores[q_idx] + top_k_scores_in_batch
                    sorted_indices = sorted(range(len(combined_scores)), key=lambda x: combined_scores[x], reverse=True)[:top_k]
                    batch_top_k_iids[q_idx] = [combined_iids[idx] for idx in sorted_indices]
                    batch_top_k_scores[q_idx] = [combined_scores[idx] for idx in sorted_indices]

        for k, q_example in enumerate(query_batch):
            q_example.retrieved_iids = batch_top_k_iids[k]
            q_example.retrieved_scores = batch_top_k_scores[k]
            eval_dataset.query_examples[i * args.batch_size + k] = q_example

            results[q_example.qid] = {
                "retrieved_iids": batch_top_k_iids[k],
                "retrieved_scores": batch_top_k_scores[k]
            }

    with open("retrieval_results.json", "w") as f:
        json.dump(results, f, indent=4)

    if args.dataset in ["fiq-dress", "fiq-shirt", "fiq-toptee", "happy"]:
        eval_dataset.evaluate_recall()
    elif args.dataset in ["circo"]:
        eval_dataset.write_to_file(os.path.join(args.output, args.dataset + "_" + args.model_size))
    else:
        raise NotImplementedError

    print("Inference Done.")
