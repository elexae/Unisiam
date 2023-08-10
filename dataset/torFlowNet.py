import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder
import random

import csv

def preprocess(traffic_feature):
    feature = traffic_feature.tolist()
    i = 0
    while i < len(feature)-1:
        if feature[i] == 1 and feature[i+1] == -1:
            if i == 0:
                if feature[i+2] == 1:
                    del feature[i]
                    del feature[i]
                    i -= 1
            elif i+1 == len(feature)-1:
                del feature[i]
                del feature[i]
                i -= 1
            else:
                if feature[i-1] == -1 and feature[i+2] == 1:
                    del feature[i]
                    del feature[i]
                    i -= 1
        i += 1
    return np.array(feature)

def resize(x, size):
    if len(x) > size:
        x = x[:size]
    else:
        x = np.hstack((x, np.array([0]*(size-len(x)))))
    return x

class torFlowNet(Dataset):
    def __init__(self, data_path, flow_size, partition='train', transform=None):
        super(Dataset, self).__init__()
        self.data_root = data_path
        self.partition = partition
        self.transform = transform
        self.size = flow_size

        file_path = os.path.join(data_path, '{}.npz'.format(self.partition))
        self.features, self.labels = self._read_data(file_path)   

    def _read_data(self, file_path):

        data = np.load(file_path, allow_pickle=True)
        
        features = data['feature']
        features = [f[f!=0] for f in features]

        labels_name = data['label']
        le = LabelEncoder()
        labels = le.fit_transform(labels_name)

        return features, labels

    def __getitem__(self, item):
        feature = self.features[item]
        if self.transform is not None:
            feature = self.transform(feature)
        target = self.labels[item]
        return feature, target

    def __len__(self):
        return len(self.labels)
