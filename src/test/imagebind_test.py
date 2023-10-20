from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import numpy.random as random
import numpy as np
import json
import os
import os.path as osp
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from utils.utils import getR1Accuary, getR5Accuary


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained='imagebind_huge.pth')
model.eval()
model.to(device)

pair = json.load(open('fscoco/test.json', 'r'))
acc = []
for i in tqdm(range(20)):
    ori_image=[]
    text_list=[]
    sketch_image=[]
    for j in range(100):
        caption_path = os.path.join('fscoco/', 'text/'+pair[i*50+j]['caption'])
        image_path = os.path.join('fscoco/', 'images/'+pair[i*50+j]['image'])
        f = open(caption_path, 'r')
        caption = f.readline().replace('\n', '')
        text_list.append(caption)
        ori_image.append(image_path)
        # sketch_image.append(osp.join('imagenet/imagenet-sketch/', other[index]['sketch_image']))
    input1 = {
        ModalityType.VISION: data.load_and_transform_vision_data(ori_image, device),
    }
    input2 = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
        # ModalityType.VISION: data.load_and_transform_vision_data(sketch_image, device),
    }

    with torch.no_grad():
        embeddings1 = model(input1)
        embeddings2 = model(input2)

    # prob = torch.softmax(embeddings1[ModalityType.VISION] @ embeddings1[ModalityType.TEXT].T, dim=-1)
    prob = torch.softmax(embeddings1[ModalityType.VISION] @ embeddings2[ModalityType.TEXT].T, dim=-1)

    acc.append(getR5Accuary(prob))  

r1 = sum(acc)/len(acc)
print(r1)