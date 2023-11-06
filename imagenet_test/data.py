import json
import os
from PIL import Image
from open_clip.factory import image_transform
from torch.utils.data import Dataset


image_mean = (0.48145466, 0.4578275, 0.40821073)
image_std = (0.26861954, 0.26130258, 0.27577711)
process_val = image_transform(224, False, image_mean, image_std)


class T2ITestDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.root_file_path = self.args.root_file_path
        self.dataset = json.load(open(self.args.root_json_path,'r'))
        self.image_transform = process_val
    

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        long_caption = self.dataset[index]['caption']
        original_image_path = os.path.join(self.root_file_path, self.dataset[index]['image_path'])
        original_image = self.image_transform(Image.open(original_image_path).convert('RGB'))
        return [original_image, long_caption]
    

class S2ITestDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.root_file_path = self.args.root_file_path
        self.other_file_path = self.args.other_file_path
        self.root_dataset = json.load(open(self.args.root_json_path,'r'))
        self.other_dataset = json.load(open(self.args.other_json_path,'r'))
        self.image_transform = process_val
    

    def __len__(self):
        return len(self.other_dataset)
    

    def __getitem__(self, index):
        original_image_path = os.path.join(self.root_file_path, self.root_dataset[index]['image'])
        original_image = self.image_transform(Image.open(original_image_path))
        original_classname = self.root_dataset[index]['classname']
        other_image_path = os.path.join(self.other_file_path, self.other_dataset[index]['image'])
        other_image = self.image_transform(Image.open(other_image_path))
        other_image_classname = self.other_dataset[index]['classname']
        return [original_image, other_image, original_classname, other_image_classname]
    

class M2ITestDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.root_file_path = self.args.root_file_path
        self.other_file_path = self.args.other_file_path
        self.dataset = json.load(open(self.args.root_json_path,'r'))
        self.image_transform = process_val
    

    def __len__(self):
        return len(self.dataset)
    

    def __getitem__(self, index):
        original_image_path = os.path.join(self.root_file_path, self.dataset[index]['image_path'])
        other_image_path = os.path.join(self.other_file_path, self.dataset[index]['image_path'])
        original_image = self.image_transform(Image.open(original_image_path))
        other_image = self.image_transform(Image.open(other_image_path))
        return [original_image, other_image]