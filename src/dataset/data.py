import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class StyleT2IDataset(Dataset):
    def __init__(self, root_path, json_path, image_transform):
        self.root_path = root_path
        self.dataset = json.load(open(json_path,'r'))
        self.image_transform = image_transform
    

    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, index):
        caption_path = os.path.join(self.root_path, 'text/'+self.dataset[index]['caption'])
        image_path = os.path.join(self.root_path, 'images/'+self.dataset[index]['image'])
        neg = np.random.randint(1, len(self.dataset))
        while neg == index:
            neg = np.random.randint(1, len(self.dataset))
        negative_path = os.path.join(self.root_path, 'images/'+self.dataset[neg]['image'])
        
        f = open(caption_path, 'r')
        caption = f.readline().replace('\n', '')
        pair_image = self.image_transform(Image.open(image_path))
        negative_image = self.image_transform(Image.open(negative_path))

        return [caption, pair_image, negative_image]
    

class StyleI2IDataset(Dataset):
    def __init__(self, root_path, json_path, image_transform):
        self.root_path = root_path
        self.dataset = json.load(open(json_path,'r'))
        self.image_transform = image_transform
    

    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, index):
        ori_path = os.path.join(self.root_path, 'images/'+self.dataset[index]['image'])
        rand = np.random.randint(1,4)
        if rand == 1:
            pair_path = os.path.join(self.root_path, 'sketch/'+self.dataset[index]['image'])
        elif rand == 2:
            pair_path = os.path.join(self.root_path, 'art/'+self.dataset[index]['image'])
        elif rand == 3:
            pair_path = os.path.join(self.root_path, 'mosaic/'+self.dataset[index]['image'])

        neg = np.random.randint(1, len(self.dataset))
        while neg == index:
            neg = np.random.randint(1, len(self.dataset))
        negative_path = os.path.join(self.root_path, 'images/'+self.dataset[neg]['image'])
        
        ori_image = self.image_transform(Image.open(ori_path))
        pair_image = self.image_transform(Image.open(pair_path))
        negative_image = self.image_transform(Image.open(negative_path))

        return [ori_image, pair_image, negative_image]


class T2ITestDataset(Dataset):
    def __init__(self, root_path, json_path, image_transform):
        self.root_path = root_path
        self.dataset = json.load(open(json_path,'r'))
        self.image_transform = image_transform
    

    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, index):
        caption_path = os.path.join(self.root_path, 'text/'+self.dataset[index]['caption'])
        image_path = os.path.join(self.root_path, 'images/'+self.dataset[index]['image'])
        
        f = open(caption_path, 'r')
        caption = f.readline().replace('\n', '')
        pair_image = self.image_transform(Image.open(image_path))

        return [caption, pair_image, index]
    

class I2ITestDataset(Dataset):
    def __init__(self, style, root_path, json_path, image_transform):
        self.style = style
        self.root_path = root_path
        self.dataset = json.load(open(json_path,'r'))
        self.image_transform = image_transform
    

    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, index):
        ori_path = os.path.join(self.root_path, 'images/'+self.dataset[index]['image'])
        pair_path = os.path.join(self.root_path, '{}/'.format(self.style)+self.dataset[index]['image'])
        
        ori_image = self.image_transform(Image.open(ori_path))
        pair_image = self.image_transform(Image.open(pair_path))

        return [ori_image, pair_image, index]
    

class X2ITestDataset(Dataset):
    def __init__(self, style, root_path, json_path, image_transform):
        self.style = style
        self.root_path = root_path
        self.dataset = json.load(open(json_path,'r'))
        self.image_transform = image_transform
    

    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, index):
        caption_path = os.path.join(self.root_path, 'text/'+self.dataset[index]['caption'])
        ori_path = os.path.join(self.root_path, 'images/'+self.dataset[index]['image'])
        pair_path = os.path.join(self.root_path, '{}/'.format(self.style)+self.dataset[index]['image'])
        
        f = open(caption_path, 'r')
        caption = f.readline().replace('\n', '')
        ori_image = self.image_transform(Image.open(ori_path))
        pair_image = self.image_transform(Image.open(pair_path))

        return [caption, ori_image, pair_image, index]
    

class VisualizationDataset(Dataset):
    def __init__(self, root_path, json_path, image_transform):
        self.root_path = root_path
        self.dataset = json.load(open(json_path,'r'))
        self.image_transform = image_transform


    def __len__(self):
        return len(self.dataset)
    
    
    def __getitem__(self, index):
        ori_path = os.path.join(self.root_path, 'images/'+self.dataset[index]['image'])
        sketch_path = os.path.join(self.root_path, 'sketch/'+self.dataset[index]['image'])
        art_path = os.path.join(self.root_path, 'art/'+self.dataset[index]['image'])
        mosaic_path = os.path.join(self.root_path, 'mosaic/'+self.dataset[index]['image'])
        
        ori_image = self.image_transform(Image.open(ori_path))
        sketch_image = self.image_transform(Image.open(sketch_path))
        art_image = self.image_transform(Image.open(art_path))
        mosaic_image = self.image_transform(Image.open(mosaic_path))

        return [ori_image, sketch_image, art_image, mosaic_image, index]