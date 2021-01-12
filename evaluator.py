# Copyright (c) Microsoft Corporation.
# Copyright (c) 2021 HongChien Yu
# Licensed under the MIT license.

from data import PRFDataset, batcher
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import logging
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler


def predict(device, model, config, args):
    total = 0
    pred_dict = dict()
    batch_size = config['training']['test_batch_size']
    fname = config["system"]["test_data"]

    logging.info(f"Evaluating {fname}")
    dataset = PRFDataset(fname, config["model"], args.tokenizer, is_test=True)
    sampler = DistributedSampler(dataset) if args.distributed else None
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler,
                            collate_fn=batcher(is_test=True), shuffle=False, num_workers=0)
    for batch in tqdm(dataloader):
        g = batch[0]
        g = g.to(device)
        g.ndata['encoding'] = g.ndata['encoding'].to(device)
        g.ndata['encoding_mask'] = g.ndata['encoding_mask'].to(device)
        g.ndata['segment_id'] = g.ndata['segment_id'].to(device)
        logits_score, logits_pred = model.network(g, device)
        logits_score = logits_score.reshape((batch_size, logits_score.shape[0] // batch_size, 1))
        logits_score = F.softmax(logits_score, dim=1)
        logits_pred = F.softmax(logits_pred, dim=1)
        logits_pred = logits_pred.reshape((batch_size, logits_pred.shape[0] // batch_size, 2))
        final_score = torch.matmul(logits_pred.permute(0, 2, 1), logits_score)
        total += 1
        qids = batch[1]
        for i, qid in enumerate(qids):
            qry_pred = pred_dict.get(qid, [])
            score = str(final_score.data.cpu().numpy()[i][1][0])
            qry_pred.append(score)
            pred_dict[qid] = qry_pred

    return pred_dict
