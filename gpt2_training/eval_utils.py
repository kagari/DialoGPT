#  Copyright (c) Microsoft Corporation. 
#  Licensed under the MIT license. 
import torch
import logging

import numpy as np

from pycocoevalcap.bleu.bleu import Bleu
from collections import defaultdict
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)

EOS_ID = 50256


def cal_BLEU_4(generated, reference, is_corpus=False):
    BLEUscore = [0.0, 0.0, 0.0, 0.0]
    for idx, g in enumerate(generated):
        if is_corpus:
            score, scores = Bleu(4).compute_score(reference, {0: [g]})
        else:
            score, scores = Bleu(4).compute_score({0: [reference[0][idx]]},
                                                  {0: [g]})
        for i, s in zip([0, 1, 2, 3], score):
            BLEUscore[i] += s
    BLEUscore[0] = BLEUscore[0]/len(generated)
    BLEUscore[1] = BLEUscore[1]/len(generated)
    BLEUscore[2] = BLEUscore[2]/len(generated)
    BLEUscore[3] = BLEUscore[3]/len(generated)
    return BLEUscore


def cal_entropy(generated):
    etp_score = [0.0, 0.0, 0.0, 0.0]
    div_score = [0.0, 0.0, 0.0, 0.0]
    counter = [defaultdict(int), defaultdict(int),
               defaultdict(int), defaultdict(int)]
    for gg in generated:
        g = gg.rstrip().split()
        for n in range(4):
            for idx in range(len(g)-n):
                ngram = ' '.join(g[idx:idx+n+1])
                counter[n][ngram] += 1
    for n in range(4):
        total = sum(counter[n].values()) + 1e-10
        for v in counter[n].values():
            etp_score[n] += - (v+0.0) / total * (np.log(v+0.0) - np.log(total))
        div_score[n] = (len(counter[n].values())+0.0) / total
    return etp_score, div_score

def forward_step(model, input_ids, position_ids, token_ids, lm_labels):
    outputs = model(input_ids=input_ids, position_ids=position_ids,
                    token_type_ids=token_ids, return_dict=True)
    lm_logits = outputs["logits"]
    # loss = F.cross_entropy(
    #     lm_logits.view(-1, lm_logits.size(-1)),
    #     label_ids.view(-1),
    #     ignore_index=tokenizer.pad_token_id,
    #     reduction="mean"
    # )
    # with torch.no_grad():
    #     ppl = loss.exp()
    loss_fct1 = CrossEntropyLoss(ignore_index=-1, reduction='none')
    loss1 = loss_fct1(lm_logits.view(-1, lm_logits.size(-1)),
                      lm_labels.view(-1))
    loss1 = loss1.view(lm_labels.size(0), lm_labels.size(1))
    label_size = torch.sum(lm_labels != -1, dim=1).type(loss1.type())
    loss = torch.sum(loss1)/torch.sum(label_size)
    ppl = torch.exp(torch.mean(torch.sum(loss1, dim=1).float()
                               / label_size.float()))
    return loss, ppl

def eval_model_loss(model, tokenizer, eval_dataloader, epoch_id, args):
    # use the same signature with eval_model_generation
    logger.info('compute eval model loss, using eval mode, '
                'please change it back to train after calling this function')
    model.eval()
    tot_loss = []
    tot_ppl = []
    tot_sample = []
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            input_ids, position_ids, token_ids, label_ids, src_len, _ = batch
            if args.no_token_id:
                token_ids = None
            n_sample = input_ids.shape[0]
            loss, ppl = forward_step(model, input_ids, position_ids, token_ids, label_ids)
            tot_loss.append(loss.mean().item() * n_sample)
            tot_ppl.append(ppl.mean().item() * n_sample)
            tot_sample.append(n_sample)
    print(f"\n Epoch {epoch_id}: Val loss {np.sum(tot_loss) / np.sum(tot_sample)} Val ppl {np.sum(tot_ppl) / np.sum(tot_sample)} ")
    return np.sum(tot_loss) / np.sum(tot_sample), np.sum(tot_ppl) / np.sum(tot_sample)
