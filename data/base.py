import torch
import torch.nn as nn
import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import cv2
import numpy as np


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, path, batch_size=1, transform=None):
        self.inputs = []
        self.labels = []
        self.transform = transform
        self.path = path
        self.batch_size = batch_size
        self.load_data()

    def load_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        input = row['input']
        label = row['label']

        if self.transform:
            input = self.transform(input)

        return input, label


class FolderDataset(BaseDataset):
    def __init__(self, path, batch_size=1, transform=None):
        super(FolderDataset, self).__init__(path, batch_size, transform)

    def load_data(self):
        self.inputs = []
        self.labels = []
        
        for label in os.listdir(self.path):
            for file in os.listdir(os.path.join(self.path, label)):
                self.inputs.append(os.path.join(self.path, label, file))
                self.labels.append(label)
        self.labels = LabelEncoder().fit_transform(np.array(self.labels).reshape(-1, 1))

        self.inputs = [self.inputs[i:i + self.batch_size] for i in range(0, len(self.inputs), self.batch_size)]
        self.labels = [self.labels[i:i + self.batch_size] for i in range(0, len(self.labels), self.batch_size)]
        self.df = pd.DataFrame({'input': self.inputs, 'label': self.labels})


class ImageFolderDataset(FolderDataset):
    def __init__(self, path, batch_size=1, transform=None):
        super(ImageFolderDataset, self).__init__(path, batch_size, transform)

    def load_data(self):
        super(ImageFolderDataset, self).load_data()
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        input = row['input']
        label = row['label']
        inputs = [cv2.imread(img) for img in input]
        
        if self.transform:
            inputs = [self.transform(img) for img in inputs]
        input = torch.stack(inputs)
        return input, label
    

class CSVDataset(BaseDataset):
    def __init__(self, path, transform=None):
        super(CSVDataset, self).__init__(path, transform)

    def load_data(self):
        self.df = pd.DataFrame()
        df = pd.read_csv(self.path)
        self.df['input'] = df[df.columns[:-1]].values.tolist()
        if isinstance(df[df.columns[-1]].iloc[0], str):
            labels = df[df.columns[-1]].astype('category').cat.codes
            labels = OneHotEncoder().fit_transform(labels.values.reshape(-1, 1)).toarray()
            self.df['label'] = labels.tolist()
        else:
            self.df['label'] = df[df.columns[-1]]
        

from sklearn.cluster import DBSCAN
class DBSCANBatchDataset(BaseDataset):
    def __init__(self, path, transform=None, batch_size=1, attrs=None, eps=0.5, min_samples=5):
        '''
        attrs: list of attributes to use for clustering
        '''
        self.batch_size = batch_size
        self.eps = eps
        self.min_samples = min_samples
        self.attrs = attrs
        super(DBSCANBatchDataset, self).__init__(path, transform)

    def load_data(self):
        self.df = pd.DataFrame()
        self.batches = {}
        df = pd.read_csv(self.path)
        self.df = df
        self.df['input'] = df[df.columns[:-1]].values.tolist()
        if isinstance(df[df.columns[-1]].iloc[0], str):
            self.df['label'] = df[df.columns[-1]].astype('category').cat.codes
        else:
            self.df['label'] = df[df.columns[-1]]
        self.df['batch_label'] = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(df[self.attrs])
        self.df['batch_label'] = self.df['batch_label'].astype(str)
        self.batches = self.df.groupby('batch_label')
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        input = row['input']
        label = row['label']
        batch_label = row['batch_label']

        if self.transform:
            input = self.transform(input)

        return input, label, batch_label