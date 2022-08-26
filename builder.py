# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim=770, K=141*20, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; K can be changed in the future work
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.dim = dim

        # create the queue
        self.register_buffer("queue", torch.randn(self.dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue




        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        #assert self.K % batch_size == 0  # for simplicity
        if ptr + batch_size >= (self.K-1):                    # move pointer
            ptr = 0

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        if ptr + batch_size >= (self.K-1):                    # move pointer
            ptr = 0
        ptr = ptr + batch_size


        self.queue_ptr[0] = ptr


        #return self.queue, self.queue_ptr[0]


    def forward(self, embedding_batch, CLabel,NumofLabel):
        """
        Input:
            CLabel: indicates if the sample actually own this label. (1 or 0)
            NumofLabel: index of the projection. for every embedding_batch it is 0...number of labels.
        Output:
            memory bank
            position of the starting for the next time storing.

        """
        #two new columns are added into the tensor so that locations of projections in label spaces can be esaily identified.
        embedding_batch_with_label_added = torch.cat((embedding_batch.view(NumofLabel, 768), CLabel.view(NumofLabel, 1).to('cuda:0')),1)

        #the following tensor is the final tensor with both label and label indexes added in
        embedding_batch = torch.cat((embedding_batch_with_label_added, torch.arange(NumofLabel).view(NumofLabel, 1).to('cuda:0')), 1)





        # dequeue and enqueue, put the new sample projections into memory bank.
        self._dequeue_and_enqueue(embedding_batch)


        return self.queue, self.queue_ptr[0]


