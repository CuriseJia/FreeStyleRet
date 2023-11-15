import os 
import numpy as np
import json
from tqdm import tqdm
from PIL import Image

path = 'DSR/'

out = 'DSR/dataset.json'

temp = []

def generate_json_from_dataset():
    folderlist = os.listdir(path+'images/')
    for folder in tqdm(folderlist):
        filelist = os.listdir(path+'images/'+folder)
        
        for file in filelist:
            ori_path = folder+'/'+file
            text_path = folder+'/'+file.split('.')[0]+'.txt'
            data = {
                'image' : ori_path,
                'caption' : text_path,
            }
            temp.append(data)

    with open(out, "w") as file:
        json.dump(temp, file)


if __name__ == '__main__':
    generate_json_from_dataset()