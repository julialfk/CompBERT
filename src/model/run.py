import argparse
import logging
import os
import json
import torch
import numpy as np
from train import TextDataset, format_json
from model import Model
from pathlib import Path
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase (a jsonl file).")  

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")  
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")     
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")      

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    #print arguments
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(f"device: {device}\n")

    # Build model
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
    model = RobertaModel.from_pretrained("microsoft/unixcoder-base")
    model = Model(model)

    # Loads fine-tuned model
    output_dir = os.path.join(args.model_name_or_path, '{}'.format("pytorch_model.bin"))
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(torch.load(output_dir, map_location=torch.device("cpu")))

    # format_json(args.eval_data_file, Path(args.eval_data_file))

    dataset = TextDataset(tokenizer, args, args.eval_data_file)
    print(f"length data: {len(dataset)}")
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, num_workers=4)

    for idx, example in enumerate(dataset.examples[:3]):
        print("*** Example ***")
        print("idx: {}".format(example.idx))
        print("code_tokens: {}".format([x.replace('\u0120', '_') for x in example.code_tokens]))
        print("code_ids: {}".format(' '.join(map(str, example.code_ids))))
        print("nl_tokens: {}".format([x.replace('\u0120', '_') for x in example.nl_tokens]))
        print("nl_ids: {}\n".format(' '.join(map(str, example.nl_ids))))

    model.eval()
    code_vecs = []
    nl_vecs = []
    label_vecs = []
    indexes = []
    for idx, batch in enumerate(dataloader):
        if idx % 50 == 0:
            print(f"current batch: {idx}")
        nl_inputs = batch["nl_input"].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs)
            nl_vecs.append(nl_vec)

        code_inputs = batch["code_input"].to(args.device)
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec)

        label_vecs.append(batch["changed"].to(args.device))
        indexes.append(batch["idx"])

    model.train()
    code_vecs = torch.cat(code_vecs, dim=0)
    nl_vecs = torch.cat(nl_vecs, dim=0)
    label_vecs = torch.cat(label_vecs, dim=0)
    indexes = torch.cat(indexes, dim=0).tolist()
    print(f"Tensor shape: {code_vecs.shape}")
    # print(f"example code_vec: {code_vecs[0]}")
    # print(f"example nl_vec: {nl_vecs[0]}")
    print(f"example label_vec: {label_vecs[0]}")


    # Calcuate scores and loss
    scores = torch.einsum("ij,ij->i", code_vecs, nl_vecs)
    probabilities = torch.special.expit(scores).tolist()
    loss_fct = BCEWithLogitsLoss()
    loss = loss_fct(scores, label_vecs.float())
    print("scores:")
    print(scores)
    print("indexes:")
    print(torch.tensor(indexes))
    print("labels:")
    print(label_vecs)

    file_path = Path(args.eval_data_file)
    with file_path.open("r") as f:
        entries = json.load(f)

    indexes_and_scores = sorted(zip(indexes, probabilities), key=lambda x: x[1], reverse=True)
    print("ordered indexes, scores:")
    print(indexes_and_scores)

    labels_sorted = [entries[idx]["changed"] for idx, _ in indexes_and_scores]
    print("labels sorted:")
    print(labels_sorted)
    print(f"loss: {loss}")


if __name__ == "__main__":
    main()
