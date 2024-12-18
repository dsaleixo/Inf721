
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DataSetFeatureExtractor(Dataset):
    
    def __init__(self,datas):
        self.datas = datas
        
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx]
    
class ReadDataFeatureExtractor():
    
    @staticmethod
    def read(device):
        datas = []
        for i in range(50):
            for j in range(50):
                loaded_data = np.load(f'datas/data_{i}_{j}.npy')
                r,c,h,w=loaded_data.shape
                loaded_data = loaded_data.reshape((r*c,h,w))
                tensor = torch.from_numpy(loaded_data).to(torch.float32)
                
                datas.append(tensor)
        return datas