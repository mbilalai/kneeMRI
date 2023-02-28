import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data
import utils as ut

class MRIDataset:
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.path[index])

        label = torch.FloatTensor([self.label[index]])
        weight = torch.FloatTensor([self.weights[self.labels[index]]])

        if self.train:
            array = ut.random_shift(array, 25)
            array = ut.random_rotate(array, 25)
            array = ut.random_flip(array, 25)
            
            array = (array - 58.09) / 49.73
            array = np.stack((array,)*3), axis=1)

            array = torch.FloatTensor(array)
            return array, label, weight
