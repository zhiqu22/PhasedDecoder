import math

from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq import metrics, utils

from collections import deque

import torch
import torch.nn as nn


@register_criterion("label_smoothed_cross_entropy_instruction")
class LabelSmoothedCrossEntropyCriterionInstruction(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False,
                 contrastive_flag="False", temperature=1.0, dim=128, mode=1, cl_position=6, contrastive_type="enc"):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.contrastive_flag = True if contrastive_flag == "True" else False
        if self.contrastive_flag:
            self.similarity = nn.CosineSimilarity(dim=-1)
            self.temperature = temperature
            self.dim = dim
            self.mode = mode
            self.cl_position = cl_position - 1
            self.contrastive_type = contrastive_type

    
    @staticmethod
    def add_args(parser):
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument("--contrastive-flag", type=str, default="False",)
        parser.add_argument("--temperature", type=float, default=1.0,)
        parser.add_argument("--dim", type=int, default=128, help="dim for cosine")
        parser.add_argument("--mode", type=int, default=1, help="1 or 4")
        # 0 is encoder's output
        parser.add_argument("--cl-position", type=int, default=6, help="layers of decoder")
        # enc dec both
        parser.add_argument("--contrastive-type", type=str, default="enc")
    
    # targets in the mode of 1:
    # y ... <eos> pad pad
    # reshape it to: pad pad <tag> y <eos>
    # <tag> can be collected from src_tokens
    def build_identity_for_1(self, sample):
        tags_indices = sample["net_input"]["src_tokens"].eq(self.padding_idx).sum(dim=1).unsqueeze(1)
        prefix = torch.gather(sample["net_input"]["src_tokens"], dim=1, index=tags_indices)
        src_tokens = sample["target"].clone()
        src_tokens = torch.cat([prefix, src_tokens], dim=1)
        src_mask = src_tokens.eq(self.padding_idx)
        src_lengths = (~src_mask).sum(dim=1)
        if src_mask.any():
            all_length = src_tokens.shape[1]
            src_indices = torch.nonzero(src_lengths < all_length).squeeze(1).tolist()
            for i in src_indices:
                padding_num = all_length - src_lengths[i].item()
                clone_sample = src_tokens[i : i+1].clone()
                src_tokens[ i : i+1, :padding_num] = clone_sample[:,all_length - padding_num: ]
                src_tokens[ i : i+1, padding_num:] = clone_sample[:, :all_length - padding_num]
        return {
            "net_input": {
                "src_tokens": src_tokens.detach().contiguous(),
                "src_lengths": src_lengths,
                "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            },
            'nsentences': sample['nsentences'],
            'ntokens': sample['ntokens'],
            "target": sample["target"],
            "id": sample["id"],
        }
    # targets in the mode of 4:
    # <tag> y ... <eos> pad pad
    # reshape it to: pad pad <tag> y <eos>
    def build_identity_for_4(self, sample):
        src_tokens = sample["target"].clone()
        src_mask = src_tokens.eq(self.padding_idx)
        src_lengths = (~src_mask).sum(dim=1)
        if src_mask.any():
            all_length = src_tokens.shape[1]
            src_indices = torch.nonzero(src_lengths < all_length).squeeze(1).tolist()
            for i in src_indices:
                padding_num = all_length - src_lengths[i].item()
                clone_sample = src_tokens[i : i+1].clone()
                src_tokens[ i : i+1, :padding_num] = clone_sample[:,all_length - padding_num: ]
                src_tokens[ i : i+1, padding_num:] = clone_sample[:, :all_length - padding_num]
        
        return {
            "net_input": {
                "src_tokens": src_tokens.detach().contiguous(),
                "src_lengths": src_lengths,
                "prev_output_tokens": sample["net_input"]["prev_output_tokens"],
            },
            'nsentences': sample['nsentences'],
            'ntokens': sample['ntokens'],
            "target": sample["target"],
            "id": sample["id"],
        }
            

    def contrastive(self, anchors, positives, real_sample):
        src_tokens = real_sample["net_input"]["src_tokens"]
        indices_tag = torch.argmax(src_tokens, dim=-1)
        tags = src_tokens[range(src_tokens.size(0)), indices_tag]

        num_total = tags.size(0)
        unique, counts = tags.unique(return_counts=True)
        counts = counts.float() / num_total
        weights = torch.zeros_like(tags, dtype=torch.float)
        for tag_tmp, count_tmp in zip(unique, counts):
            p_tmp = 1 - count_tmp
            weights[tags == tag_tmp] = p_tmp

        batch_size = anchors.shape[0]
        # 利用real_sample做负例
        positive_similarities = self.similarity(anchors, positives)
        expanded_anchors = anchors.unsqueeze(1).repeat(1, batch_size, 1)
        negatives = expanded_anchors.transpose(0, 1)
        negative_similarities = self.similarity(expanded_anchors, negatives)
        negative_similarities[torch.arange(batch_size), torch.arange(batch_size)] = positive_similarities
        loss = -nn.LogSoftmax(0)(torch.div(negative_similarities, self.temperature)).diag()
        loss = loss * weights
        return loss.sum()
    


    def forward(self, model, sample, reduce=True):
        contrastive_loss = 0
        
        if self.contrastive_flag:
            if self.contrastive_type == "dec" or self.contrastive_type == "both":
                assert self.mode == 4, "mode should be 4 when using dec and both for contrastive learning."
            net_output = model(**sample["net_input"], contrastive_flag=True, contrastive_position=self.cl_position, identity_flag=False)
            with torch.no_grad():
                if self.mode == 1: 
                    identity_sample = self.build_identity_for_1(sample)
                else:
                    identity_sample = self.build_identity_for_4(sample)
            identity_output = model(**identity_sample["net_input"], contrastive_flag=True, contrastive_position=self.cl_position, identity_flag=True)
            if self.contrastive_type == "enc":
                contrastive_loss = self.contrastive(net_output[1]["src_instructions"][:,:self.dim], identity_output[1]["src_instructions"][:,:self.dim], sample)
            else:
                assert net_output[1]["tgt_instructions"] is not None, "representation is lacking when using dec and both for contrastive learning."
                if self.contrastive_type == "both":
                    contrastive_loss = self.contrastive(net_output[1]["src_instructions"][:,:self.dim], identity_output[1]["src_instructions"][:,:self.dim], sample) + \
                                           self.contrastive(net_output[1]["tgt_instructions"][:,:self.dim], identity_output[1]["tgt_instructions"][:,:self.dim], sample)
                if self.contrastive_type == "dec":
                    contrastive_loss = self.contrastive(net_output[1]["tgt_instructions"][:,:self.dim], identity_output[1]["tgt_instructions"][:,:self.dim], sample)
        else:
            net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        nsentences = sample["target"].size(0)
        ntokens = sample["ntokens"]
        n_src_tokens = sample["net_input"]["src_lengths"].sum().item()
        all_loss = loss + contrastive_loss * n_src_tokens / nsentences
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        if isinstance(contrastive_loss, int):
            logging_output["contrastive_loss"] = 0
        else:
            logging_output["contrastive_loss"] = utils.item(contrastive_loss.data)
        
        return all_loss, sample_size, logging_output
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        super().reduce_metrics(logging_outputs)
        nsentences = utils.item(
            sum(log.get("nsentences", 0) for log in logging_outputs)
        )
        contrastive_loss = utils.item(
            sum(log.get("contrastive_loss", 0) for log in logging_outputs)
        )
        metrics.log_scalar(
            "contrastive_loss",
            contrastive_loss / nsentences / math.log(2),
            nsentences,
            round=3,
        )