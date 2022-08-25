import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F




# class NTXent(nn.Module):
#
#     def __init__(self, config, tau=1.):
#         super(NTXent, self).__init__()
#         self.tau = tau
#         self.norm = 1.
#         self.transform = nn.Sequential(
#             nn.Dropout(config.hidden_dropout_prob),
#             nn.Linear(config.hidden_size, config.hidden_size),
#             nn.ReLU(),
#             nn.Dropout(config.hidden_dropout_prob),
#             nn.Linear(config.hidden_size, config.hidden_size),
#         )
#
#     def forward(self, x, labels=None):
#
#         x = self.transform(x)
#
#         n = x.shape[0]
#         x = F.normalize(x, p=2, dim=0) / np.sqrt(self.tau)
#         # 2B * 2B
#         sim = x @ x.t()
#
#
#         sim[np.arange(n), np.arange(n)] = -1e9
#
#         logprob = F.log_softmax(sim, dim=1)
#
#         m = 2
#
#         labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
#         # remove labels pointet to itself, i.e. (i, i)
#         labels = labels.reshape(n, m)[:, 1:].reshape(-1)
#         loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1) / self.norm
#
#         return loss




class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        features = F.normalize(features, dim=1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)


        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()


        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) #/ mask.sum(1)
        # print("mean_log_prob_pos")
        # print(mean_log_prob_pos)


        # loss
        loss = -mean_log_prob_pos
        loss = loss.nansum() / anchor_dot_contrast.shape[0]
        #loss = loss.nanmean()

        if torch.isnan(loss):
            print("Nan loss.")

        return loss


class contrastLoss(nn.Module):
    def __init__(self,config):
        super(contrastLoss, self).__init__()

        #self.contrast_lossfuc = NTXent(config)
        self.supcontrast_loss = SupConLoss()




    def forward(self, batch,the_memoryBank,ifTorFSample,num_label,position_indicator,memoryBANKCounter):
        # args:
        #     ifTorFSample is the ground truth label
        #     batch is the projections of samples
        #     the_memoryBank is the memory bank
        #     num_label is number of labels
        #     position_indicator is used for the case that memory bank is still not fully filled with projections
        #     memoryBANKCounter tells how many samples have been stored into the memory bank.


        final_loss=0
        i=0
        ifTorFSample_reshaped = ifTorFSample.view(num_label, -1)
        temporal = torch.cat((batch, ifTorFSample_reshaped), dim=1)
        #for loop for num_label is used for: takes the i-th projection of the current batch and builds input with other projections into i-th label space from memory bank for the contrastive loss.
        if memoryBANKCounter <= the_memoryBank.shape[1]/num_label:
            for i in range(num_label):
                #takes all the projections in i-th label space from memory bank out
                #if the memory bank is not fully filled with projections, the search of the projections should end at position_indicator
                same_label_from_memorybank = ((the_memoryBank[:position_indicator, -1] == i).nonzero(as_tuple=True)[0])
                extracted_samples_from_memorybank = torch.index_select(the_memoryBank, 0,same_label_from_memorybank)


                #build input for the contrastive loss
                pairs_for_contrastive = torch.vstack((temporal[i], extracted_samples_from_memorybank[:, :-1]))


                suploss = self.supcontrast_loss(pairs_for_contrastive[:, :-1], pairs_for_contrastive[:, -1])


                if not torch.isnan(suploss):
                    final_loss = final_loss + suploss



        else:
            for i in range(num_label):
                # takes all the projections in i-th label space from memory bank out
                same_label_from_memorybank = ((the_memoryBank[:, -1] == i).nonzero(as_tuple=True)[0])
                extracted_samples_from_memorybank = torch.index_select(the_memoryBank, 0, same_label_from_memorybank)

                # build input for the contrastive loss
                pairs_for_contrastive = torch.vstack((temporal[i], extracted_samples_from_memorybank[:, :-1]))

                suploss = self.supcontrast_loss(pairs_for_contrastive[:, :-1], pairs_for_contrastive[:, -1])
                if not torch.isnan(suploss):
                    final_loss = final_loss + suploss

        return final_loss