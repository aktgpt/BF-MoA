from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, features1, features2, neg_features=None, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        batch_size = features1.shape[0]
        device = features1.device

        # unique_classes, counts = torch.unique(labels, return_counts=True)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        self_con_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0,
        )
        positive_mask = mask  # * self_con_mask
        negative_mask = 1 - positive_mask  # * self_con_mask

        logits = F.cosine_similarity(
            F.normalize(features1, dim=1).unsqueeze(1),
            F.normalize(features2, dim=1).unsqueeze(0),
            dim=-1,
        )
        logits /= self.temperature

        if neg_features is not None:
            all_positive_mask = torch.cat([positive_mask, torch.zeros_like(positive_mask)], dim=1)
            all_negative_mask = torch.cat([negative_mask, mask], dim=1)
            neg_logits = F.cosine_similarity(
                F.normalize(features1, dim=1).unsqueeze(1),
                F.normalize(neg_features, dim=1).unsqueeze(0),
                dim=-1,
            )
            neg_logits /= self.temperature
            all_logits = torch.cat([logits, neg_logits], dim=1)
        else:
            all_positive_mask = positive_mask
            all_negative_mask = negative_mask
            all_logits = logits

        all_logits = all_logits - all_logits.max(dim=1, keepdim=True)[0].detach()

        exp_logits = torch.exp(all_logits)
        denominator = torch.sum(exp_logits * all_positive_mask, axis=1, keepdim=True) + torch.sum(
            exp_logits * all_negative_mask, axis=1, keepdim=True
        )

        log_prob = (all_logits - torch.log(denominator)) * all_positive_mask
        sum_positive_mask = torch.sum(all_positive_mask, dim=1)
        log_prob = (
            torch.sum(log_prob, dim=1)[torch.nonzero(sum_positive_mask)]
            / sum_positive_mask[torch.nonzero(sum_positive_mask)]
        )
        # if torch.sum(torch.isnan(log_prob)) > 0:
        #     # torch.sum(log_prob, dim=1)
        #     print(torch.sum(positive_mask, dim=1))
        # else:
        loss = -(self.temperature * log_prob).mean()
        return loss
