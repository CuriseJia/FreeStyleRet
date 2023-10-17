import os
import json
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader


class StyleDataset(Dataset):
    def __init__(self, root_path, json_path, image_transform, tokenizer):
        self.root_path = root_path
        self.dataset = json.load(open(json_path,'r'))
        self.image_transform = image_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return 