
# coding: utf-8

import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

class BrainDataset(Dataset):
    def __init__(self, file_list, label_list, is_train=True):
        self.is_train = is_train
        self.input_shape = 15
        self.file_list = file_list
        self.label_list = label_list
        
    def __getitem__(self, index):
        img = self.normalize_data(nib.load(self.file_list[index]).get_data())
        target = np.argmax(self.label_list[index])
        return img, target
    
    def __len__(self):
        return len(self.label_list)
    
    def normalize_data(self, data):
        if self.is_train:
            select_idx = np.random.randint(low=self.input_shape, high=data.shape[-1], size=1)[0]
            data = data[:, :, :, select_idx-self.input_shape:select_idx]
        else:
            data = data[:, :, :, :self.input_shape]
        data_mask = data.copy()
        data_mask[data_mask!=0] = 1
        mean = data.sum()/data_mask.sum()
        data = data/mean
        data[~ np.isfinite(data)] = 0
        data = data.transpose(3, 0, 1, 2)
        data = data[np.newaxis,:]
        return data

