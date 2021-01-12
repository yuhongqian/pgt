# Copyright (c) Microsoft Corporation.
# Copyright (c) 2021 HongChien Yu
# Licensed under the MIT license.

import os
import argparse
import json
import logging
import random
import numpy as np
import torch
import torch.distributed as dist
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers import AdamW, WarmupLinearSchedule
from trainer import train
from evaluator import predict
from model import PGTModel
from data import PRFDataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", "--cf", type=str, required=True,
                        help="pointer to the configuration file of the experiment")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--checkpoint', type=int, default=2500)
    parser.add_argument('--train', default=False, action='store_true', help="Whether on train mode")
    parser.add_argument('--test', default=False, action='store_true', help="Whether on test mode")
    parser.add_argument('--load_train', default=False, action='store_true', help="Load train data into cache file")
    parser.add_argument('--load_test', default=False, action='store_true', help="Load test data into cache file")
    parser.add_argument("--data_path", type=str, help="the data path to load data from.")
    parser.add_argument("--model_path", type=str, default=None, help="path to the model to test")
    parser.add_argument("--test_output", type=str, default="", help="path to store the test results")
    parser.add_argument("--distributed", default=False, action="store_true", help="Use distributed training if set")
    return parser.parse_args()


def main():
    # Load args and config
    args = parse_args()
    config = json.load(open(args.config_file, 'r', encoding="utf-8"))
    logging.info("========== Model Configuration ==========")
    logging.info(config)
    args.config = config
    args.max_seq_length = config["model"]["bert_max_len"]
    os.makedirs(config["training"]["output_dir"], exist_ok=True)

    if args.load_train or args.load_test:
        tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"])
        _ = PRFDataset(args.data_path, config["model"], bert_tokenizer=tokenizer, is_test=args.load_test, loading=True)
        return

    # Set distributed training
    if args.distributed:
        torch.distributed.init_process_group(backend="nccl")
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)

    # Set random seed for the ease of reproduction
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Loading tokenizer
    tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"])
    args.tokenizer = tokenizer

    # initialize model
    device = torch.device("cuda", local_rank) if args.distributed else torch.device("cuda")
    model = PGTModel(args, config, device)
    model.network = model.network.to(device)

    if args.model_path is not None:
        model.load(args.model_path)

    if args.test:
        logging.info("========== Testing ==========")
        model.eval()
        with torch.no_grad():
            final_pred = predict(device, model, config, args)
        json.dump(final_pred, open(args.test_output, "w"))

    # Model Training
    if args.train:
        # Prepare Optimizer
        logging.info("========== Training ==========")
        train_setting = config["training"]
        lr = train_setting["learning_rate"]
        epochs = train_setting["epochs"]

        param_optimizer = list(model.network.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        total_steps = int(train_setting["total_training_examples"] * epochs / train_setting["train_batch_size"])
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_setting["warmup_proportion"],
                                         t_total=total_steps)

        for index in range(epochs):
            train(model, device, config, args, optimizer, scheduler, epoch=index)


if __name__ == '__main__':
    main()

