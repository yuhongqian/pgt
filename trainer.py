# Copyright (c) Microsoft Corporation.
# Copyright (c) 2021 HongChien Yu
# Licensed under the MIT license.

import os
import random
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from data import PRFDataset, batcher


def train(model, device, config, args, optimizer, scheduler, epoch=0):
    model.train()
    train_files = [fname for fname in os.listdir(config["system"]['train_data']) if not fname.endswith(".cache")]
    logging.info(f"train_files = {train_files}")
    random.shuffle(train_files)
    scaler = torch.cuda.amp.GradScaler()
    batch_size = config['training']['train_batch_size']
    logging_step = config["training"]["logging_step"]
    for fname in train_files:
        logging.info(f"training file {fname}")
        dataset = PRFDataset(os.path.join(config["system"]['train_data'], fname), config["model"], args.tokenizer)

        sampler = DistributedSampler(dataset) if args.distributed else None
        dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, sampler=sampler,
                                collate_fn=batcher(), num_workers=0, drop_last=True)
        print_loss = 0
        criterion = CrossEntropyLoss()
        bce_loss_logits = nn.BCEWithLogitsLoss()

        for step, batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                g = batch[0]
                g = g.to(device)
                g.ndata['encoding'] = g.ndata['encoding'].to(device)
                g.ndata['encoding_mask'] = g.ndata['encoding_mask'].to(device)
                g.ndata['segment_id'] = g.ndata['segment_id'].to(device)
                logits_score, logits_pred = model.network(g, device)
                node_labels = batch[1]
                node_labels = node_labels.to(device)
                node_loss = bce_loss_logits(logits_score, node_labels)
                logits_score = logits_score.reshape((batch_size, logits_score.shape[0] // batch_size, 1))
                logits_score = F.softmax(logits_score, dim=1)
                logits_pred = F.softmax(logits_pred, dim=1)
                logits_pred = logits_pred.reshape((batch_size, logits_pred.shape[0] // batch_size, 2))
                final_score = torch.squeeze(torch.matmul(logits_pred.permute(0, 2, 1), logits_score), 2)
                labels = torch.squeeze(torch.tensor(batch[2], dtype=torch.long).reshape((batch_size, 1)), 1)
                labels = labels.to(device)
                pred_loss = criterion(final_score, labels)

            loss = pred_loss + node_loss
            print_loss += loss.data.cpu().numpy()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if (step + 1) % logging_step == 0 and (not args.distributed or dist.get_rank() == 0):
                logging.info("********* loss {} ************".format(print_loss/logging_step))
                logging.info(f"lr = {optimizer.param_groups[0]['lr']}")
                print_loss = 0

            if (step + 1) % args.checkpoint == 0 and (not args.distributed or dist.get_rank() == 0):
                os.makedirs(os.path.join(config['training']['output_dir']), exist_ok=True)
                model_saving_path = f"{config['training']['output_dir']}/{cache_path}_{step+1}_epoch{epoch}.pt"
                model.save(model_saving_path)
                torch.save(optimizer.state_dict(), f"{model_saving_path}.optim")
                torch.save(scheduler.state_dict(), f"{model_saving_path}.scheduler")
            model.train()
        del dataset, dataloader

    if not args.distributed or dist.get_rank() == 0:
        model_saving_path = f"{config['training']['output_dir']}/epoch{epoch}_final.pt"
        model.save(model_saving_path)
        torch.save(optimizer.state_dict(), f"{model_saving_path}.optim")
        torch.save(scheduler.state_dict(), f"{model_saving_path}.scheduler")
