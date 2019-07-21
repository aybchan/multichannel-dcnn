import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
import torch
from torch.utils.data import Dataset


class Data(Dataset):
    """

    Implements a PyTorch Dataset class

    """
    def __init__(self,dataset,testing,time_steps,dimensions):

        x = np.empty((0,time_steps))

        for dim in range(1,dimensions + 1):
            train_data = pd.DataFrame(loadarff('data/{0}Dimension{1}_TRAIN.arff'
                                               .format(dataset,dim))[0])
            dtypes = {i:np.float32 for i in train_data.columns[:-1]}
            dtypes.update({train_data.columns[-1]: np.int})
            train_data = train_data.astype(dtypes)
            x = np.vstack([x,train_data.values[:,:-1].reshape(-1,time_steps)])
            
        self.mean = np.mean(x.reshape(-1,dimensions,time_steps),axis=0)
        self.std  = np.std(x.reshape(-1,dimensions,time_steps),axis=0)
        
        self.x = (x.reshape(-1,dimensions,time_steps) - self.mean) / self.std
        self.y = train_data.iloc[:,-1].values
        
        if testing:
            x = np.empty((0,time_steps))
            for dim in range(1,dimensions + 1):
                test_data = pd.DataFrame(loadarff('data/{0}Dimension{1}_TEST.arff'
                                                   .format(dataset,dim))[0])
                dtypes = {i:np.float32 for i in test_data.columns[:-1]}
                dtypes.update({test_data.columns[-1]: np.int})
                test_data = test_data.astype(dtypes)
                x = np.vstack([x,test_data.values[:,:-1].reshape(-1,time_steps)])
                self.y = test_data.iloc[:,-1].values


        self.x = (x.reshape(-1,dimensions,time_steps) - self.mean) / self.std

        class_labels = np.unique(self.y)
        if (class_labels != np.arange(len(class_labels))).any():
            new_labels = {old_label: i for i,old_label in enumerate(class_labels)}
            self.y = [new_labels[old_label] for old_label in self.y]

    def __len__(self):
        return len(self.y)

    def __getitem__(self,idx):
        x = torch.Tensor([self.x[idx]]).float()
        y = torch.Tensor([self.y[idx]]).long()
        return x,y
