# coding=utf-8
import copy
import random
from collections import Sized
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Sampler


class FixedSizeCategoricalSampler(Sampler):
    # stolen from https://github.com/fiveai/on-episodes-fsl/blob/master/src/datasets/sampler.py
    def __init__(self,
                 data_source: Sized,
                 iterations: int,
                 n_way: int,
                 k_shot: int,
                 n_query: int,
                 replacement: Optional[bool] = True):
        super(FixedSizeCategoricalSampler, self).__init__(data_source)

        self.iterations = iterations
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.replacement = replacement
        label = np.array(data_source['label'])
        unique = np.unique(label)
        unique = np.sort(unique)

        self.index_per_label = []
        self.labels = unique
        self.index_label_map = {}

        for i in unique:
            index_for_class_i = np.argwhere(label == i).reshape(-1)
            index_for_class_i = torch.from_numpy(index_for_class_i)
            self.index_per_label.append(index_for_class_i)
            self.index_label_map[i] = list(index_for_class_i.numpy())

    def __len__(self):
        return self.iterations

    def __iter__(self):
        if self.replacement:
            for i in range(self.iterations):
                batch_gallery = []
                batch_query = []
                classes = torch.randperm(len(self.index_per_label))[:self.n_way]
                for c in classes:
                    index_for_class_c = self.index_per_label[c.item()]
                    random_indices = torch.randperm(index_for_class_c.size()[0])
                    batch_gallery.append(index_for_class_c[random_indices[:self.k_shot]])
                    batch_query.append(index_for_class_c[random_indices[self.k_shot:self.k_shot + self.n_query]])
                batch = torch.cat(batch_gallery + batch_query)
                yield batch

        else:
            n_to_sample = (self.n_query + self.k_shot)
            batch_size = self.n_way*(self.n_query + self.k_shot)

            remaining_classes = list(self.labels)

            copy_index_label_map = copy.deepcopy(self.index_label_map)

            while len(remaining_classes) > self.n_way - 1:
                # randomly select classes
                classes = random.sample(remaining_classes, self.n_way)

                batch_gallery = []
                batch_query = []

                # construct the batch
                for c in classes:
                    # sample correct numbers
                    l = random.sample(copy_index_label_map[c], n_to_sample)
                    batch_gallery.append(torch.tensor(l[:self.k_shot], dtype=torch.int32))
                    batch_query.append(torch.tensor(l[self.k_shot:self.k_shot + self.n_query],
                                                    dtype=torch.int32))

                    # remove values if used (sampling without replacement)
                    for value in l:
                        copy_index_label_map[c].remove(value)

                    # if not enough elements remain,
                    # remove key from dictionary and remaining classes
                    if len(copy_index_label_map[c]) < n_to_sample:
                        del copy_index_label_map[c]
                        remaining_classes.remove(c)

                batch = torch.cat(batch_gallery + batch_query)
                yield batch