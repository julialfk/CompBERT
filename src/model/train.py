import argparse
import logging
import os
import random
import re
import json
import torch
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
from model import Model
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 changed,
                 idx):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.changed = changed
        self.idx = idx


def convert_examples_to_features(entry, idx, tokenizer, args):
    """convert examples to token ids"""
    code_tokens = tokenizer.tokenize(entry["code"])
    nl_tokens = tokenizer.tokenize(entry["nl_input"])
    # print(f"code: {len(code_tokens)} - nl: {len(nl_tokens)}")
    if (len(code_tokens) > args.code_length-4
            or len(nl_tokens) > args.nl_length-4):
        return None

    code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    nl_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, entry["changed"], idx)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []

        file_path = Path(file_path)
        with file_path.open("r") as f:
            entries = json.load(f)

        logger.info(f"Dataset size at start: {len(entries)}")

        large_examples = 0
        issue = ""
        positive_examples = []
        negative_examples = []
        last_idx = len(entries) - 1
        # print(f"last_idx = {last_idx}")
        for idx, entry in enumerate(entries):
            # print(idx)
            if issue != entry["issue"] or idx == last_idx:
                # print(issue != entry["issue"])
                # print(f"entry: {[ord(c) for c in entry["issue"]]}")
                # print(issue is not entry["issue"])
                # Balance the positive and negative examples from the previous
                # issue and add them to the dataset
                len_positives = len(positive_examples)
                len_negatives = len(negative_examples)
                if len_positives < len_negatives:
                    np.random.shuffle(negative_examples)
                    negative_examples = negative_examples[:len_positives]
                elif len_positives > len_negatives:
                    np.random.shuffle(positive_examples)
                    positive_examples = positive_examples[:len_negatives]

                assert len(positive_examples) == len(negative_examples)
                self.examples += positive_examples + negative_examples

                positive_examples = []
                negative_examples = []
                issue = entry["issue"]

            feature = convert_examples_to_features(entry, idx, tokenizer, args)
            if feature is not None:
                if entry["changed"]:
                    positive_examples.append(feature)
                else:
                    negative_examples.append(feature)
            else:
                large_examples += 1

        logger.info(f"Dataset size after balancing: {len(self.examples)}")
        logger.info(f"Removed entries due to exceeding token limit: {large_examples}")

        # if "train" in file_path:
        #     for idx, example in enumerate(self.examples[:3]):
        #         logger.info("*** Example ***")
        #         logger.info(f"idx: {idx}")
        #         logger.info(f"code_tokens: {[x.replace('\u0120', '_') for x in example.code_tokens]}")
        #         logger.info(f"code_ids: {' '.join(map(str, example.code_ids))}")
        #         logger.info(f"nl_tokens: {[x.replace('\u0120', '_') for x in example.nl_tokens]}")
        #         logger.info(f"nl_ids: {' '.join(map(str, example.nl_ids))}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {"code_input": torch.tensor(example.code_ids),
                "nl_input": torch.tensor(example.nl_ids),
                "changed": torch.tensor(example.changed),
                "idx": example.idx}


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def format_json(input_path, output_path):
    input_path = Path(input_path)
    with input_path.open("r") as f:
        entries = json.load(f)

    for entry in entries:
        # Merge the summary and description
        summary = re.sub(r"^\[[A-Z]+\-[0-9]+\]", "", entry["summary"])
        description_soup = BeautifulSoup(entry["description"], "html.parser")
        for div in description_soup.find_all("div", {"class": "code panel"}):
            div.decompose()
        nl_input = summary + "\n" + description_soup.get_text()
        entry["nl_input"] = nl_input

    with output_path.open("w") as f:
        f.write(json.dumps(entries))

    return entries


def train(args, model, tokenizer):
    """ Train the model """
    # Get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)
    
    # Get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * args.num_train_epochs)


    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = args.num_train_epochs")
    # logger.info(f"  Instantaneous batch size per GPU = {args.train_batch_size // args.n_gpu}")
    logger.info(f"  Total train batch size  = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {len(train_dataloader) * args.num_train_epochs}")

    model.zero_grad()

    model.train()
    tr_num, tr_loss, best_mrr = 0, 0, 0
    for idx in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # Get inputs
            code_inputs = batch["code_input"].to(args.device)
            nl_inputs = batch["nl_input"].to(args.device)
            labels = batch["changed"].to(args.device)

            code_vecs = model(code_inputs=code_inputs)
            nl_vecs = model(nl_inputs=nl_inputs)

            # Calcuate scores and loss
            scores = torch.sum(code_vecs * nl_vecs, dim=1)
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(scores, labels.float())

            # Report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%100 == 0:
                logger.info(f"epoch {idx} step {step+1} loss {round(tr_loss/tr_num, 5)}")
                tr_loss = 0
                tr_num = 0

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Evaluate
        results = evaluate(args, model, tokenizer, args.eval_data_file, eval_when_training=True)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value, 4))    
            
        #save best model
        logger.info(results['eval_mrr'])
        # if results['eval_mrr']>best_mrr:
        #     best_mrr = results['eval_mrr']
        #     logger.info("  "+"*"*20)  
        #     logger.info("  Best mrr:%s",round(best_mrr, 4))
        #     logger.info("  "+"*"*20)

        checkpoint_prefix = 'checkpoint-best-mrr'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
        torch.save(model_to_save.state_dict(), output_dir)
        logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, file_name, eval_when_training=False):
    eval_dataset = TextDataset(tokenizer, args, file_name)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info(f"  Num queries = {len(eval_dataset)}")
    logger.info(f"  Batch size = {args.eval_batch_size}")

    model.eval()
    code_vecs = []
    nl_vecs = []

    for step, batch in enumerate(train_dataloader):
        # Get inputs
        code_inputs = batch["code_input"].to(args.device)
        nl_inputs = batch["nl_input"].to(args.device)
        labels = batch["changed"].to(args.device)

        code_vecs = model(code_inputs=code_inputs)
        nl_vecs = model(nl_inputs=nl_inputs)

        # Calcuate scores and loss
        scores = torch.sum(code_vecs * nl_vecs, dim=1)
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(scores, labels.float())

    # TODO: develop the validation metric

    # for batch in query_dataloader:
    #     nl_inputs = batch[1].to(args.device)
    #     with torch.no_grad():
    #         nl_vec = model(nl_inputs=nl_inputs) 
    #         nl_vecs.append(nl_vec.cpu().numpy()) 

    # for batch in code_dataloader:
    #     code_inputs = batch[0].to(args.device)    
    #     with torch.no_grad():
    #         code_vec = model(code_inputs=code_inputs)
    #         code_vecs.append(code_vec.cpu().numpy())  
    # model.train()    
    # code_vecs = np.concatenate(code_vecs,0)
    # nl_vecs = np.concatenate(nl_vecs,0)
    
    # scores = np.matmul(nl_vecs,code_vecs.T)
    
    # sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
    # nl_urls = []
    # code_urls = []
    # for example in query_dataset.examples:
    #     nl_urls.append(example.url)
        
    # for example in code_dataset.examples:
    #     code_urls.append(example.url)

    # ranks = []
    # for url, sort_id in zip(nl_urls,sort_ids):
    #     rank = 0
    #     find = False
    #     for idx in sort_id[:1000]:
    #         if find is False:
    #             rank += 1
    #         if code_urls[idx] == url:
    #             find = True
    #     if find:
    #         ranks.append(1/rank)
    #     else:
    #         ranks.append(0)

    # result = {
    #     "eval_mrr":float(np.mean(ranks))
    # }

    return result



def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, 
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
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

    # print arguments
    args = parser.parse_args()
    #set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    #set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    #build model
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    # config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained(args.model_name_or_path)

    model = Model(model)
    logger.info("Training/evaluation parameters %s", args)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Training
    if args.do_train:
        train(args, model, tokenizer)


    # output_path = Path(__file__).parent.joinpath(args.output_dir, "data_formatted.json")
    # format_json(args, output_path)



if __name__ == "__main__":
    main()