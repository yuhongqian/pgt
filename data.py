# Copyright (c) Microsoft Corporation.
# Copyright (c) 2021 HongChien Yu
# Licensed under the MIT license.

import os
import json
import logging
import torch
import dgl
from dgl import DGLGraph
from torch.utils.data import Dataset
from utils import truncate_input_sequence


def batcher(is_test=False):
    """
    collate_fn function used for the DataLoader
    :return: the collated batch
    """
    def batcher_helper(batch):
        graphs = [x[0] for x in batch]
        batch_graphs = dgl.batch(graphs)
        qid = [x[1] for x in batch]
        if not is_test:
            label = [x[2] for x in batch]
            return batch_graphs, batch_graphs.ndata['label'], label, qid
        else:
            return batch_graphs, qid
    return batcher_helper


def encode_sequence(query, candidate, passage, max_seq_len, tokenizer):
    """
    :param query: raw query string
    :param candidate: byte-pair-encoded candidate passage
    :param passage: byte-pair-encoded feedback passage
    :param max_seq_len: max sequence length of the encoder
    :param tokenizer: BERT tokenizer by default
    :return: (input token ids, input masks, sequence/segment ids)
    """
    seqA = tokenizer.tokenize(query)
    if candidate is not None:
        seqA = seqA + ["[SEP]"] + candidate
    seqA = ["[CLS]"] + seqA + ["[SEP]"]
    if passage is None:
        truncate_input_sequence(seqA, None, max_seq_len)
        input_ids = tokenizer.convert_tokens_to_ids(seqA)
        sequence_ids = [0] * len(seqA)
        input_mask = [1] * len(input_ids)
    else:
        seqB = passage + ["[SEP]"]
        truncate_input_sequence(seqA, seqB, max_seq_len)
        input_tokens = seqA + seqB
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        sequence_ids = [0] * len(seqA) + [1] * len(seqB)
        input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        sequence_ids.append(0)
        input_mask.append(0)
    return torch.LongTensor(input_ids), torch.LongTensor(input_mask), torch.LongTensor(sequence_ids)


'''
Vectorize each data point
'''
def batch_transform_bert(inst, config_model, bert_tokenizer, is_test=False):
    """
    Creates a batch of data.
    :param inst: one line from the .json input file
    :param config_model: config["model"]
    :param bert_tokenizer: BERT tokenizer by default
    :param is_test: True if this is test data
    :return: (input graph, qid) or (input graph, qid, relevance label)
    """

    bert_max_len = config_model['bert_max_len']
    # ablation:
    # base: PGT base
    # exp1: remove prepended candidate (base w/o pre dc)
    # exp2: remove prepended qry and candidate (base w/o pre q, dc)
    # exp3: remove candidate from the node  (base w/o node dc)
    # exp4: remove the entire (qry, candidate) node    (base w/o node q, dc)
    ablation = config_model["ablation"]
    num_nodes = len(inst['node'])
    if ablation != "exp4":
        num_nodes += 1     # +1 for the candidate document node

    g = DGLGraph()
    g.add_nodes(num_nodes)

    for i in range(len(inst['node'])):
        for j in range(len(inst['node'])):
            if i != j:
                g.add_edge(i, j)

    for i in range(num_nodes):
        query = inst['query']
        candidate = inst['candidate']
        if ablation != "exp4" and i == 0:  # candidate node in base, exp1, 2, 3:
            if ablation == "exp3":
                candidate = None
            if not is_test:
                g.nodes[i].data['label'] = torch.tensor(inst['label']).unsqueeze(0).type(torch.FloatTensor)
            encoding_inputs, encoding_masks, encoding_ids = encode_sequence(query, candidate, None,
                                                                                bert_max_len,
                                                                                bert_tokenizer)
        else:  # feedback nodes
            node_idx = i
            if ablation != "exp4":
                node_idx = i - 1
            elif i == 0 and not is_test:
                g.nodes[i].data['label'] = torch.tensor(inst['label']).unsqueeze(0).type(torch.FloatTensor)
            passage = inst['node'][node_idx]['passage']
            if ablation == "exp1":
                candidate = None
            elif ablation == "exp2":
                candidate, query = None, None
            encoding_inputs, encoding_masks, encoding_ids = encode_sequence(query, candidate, passage, bert_max_len,
                                                                                bert_tokenizer)

        g.nodes[i].data['encoding'] = encoding_inputs.unsqueeze(0)
        g.nodes[i].data['encoding_mask'] = encoding_masks.unsqueeze(0)
        g.nodes[i].data['segment_id'] = encoding_ids.unsqueeze(0)

    if is_test:
        return g, inst['qid']
    else:
        return g, inst['qid'], inst["label"]    # type(inst["label"]) == int


class PRFDataset(Dataset):
    def __init__(self, filename, config_model, bert_tokenizer=None, is_test=False, loading=False):
        """
        Initializes the dataset. Saves the tokenized data into .cache file if not already.
        :param filename: path to the .json input file
        :param config_model: model["config"]
        :param bert_tokenizer: BERT tokenizer by default
        :param is_test: True if this is test data
        :param loading: True if the software is in the loading mode
        """

        ablation = config_model["ablation"]
        cache_path = f"{filename}.{ablation}.cache"

        if loading and filename.endswith(".cache"):
            return
        if os.path.exists(cache_path):
            logging.info(f"Loading data from cache {cache_path}.")
            if not loading:
                self.data = torch.load(cache_path)
        else:
            logging.info(f"Processing data from file {filename}.")
            self.data = []
            with open(filename, "r") as f:
                for i, l in enumerate(f):
                    if i % 10000 == 0:
                        logging.info(f"loading example {i}")
                    example = json.loads(l)
                    encoded_example = batch_transform_bert(example, config_model, bert_tokenizer, is_test)
                    self.data.append(encoded_example)
                torch.save(self.data, cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
