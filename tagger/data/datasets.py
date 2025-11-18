from torch.utils.data import Dataset
import numpy as np


class ConstituentsDataset(Dataset):
    """Simple Dataset wrapper for constituent arrays.

    Expects X: numpy array shaped (N, n_nodes, features)
            y: numpy array shaped (N, n_classes) or (N,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        x = self.X[idx]
        label = self.y[idx]
        return (x, label)
    



    
