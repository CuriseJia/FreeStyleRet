from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import json
import os
from tqdm import tqdm

from utils.utils import getR1Accuary, getR5Accuary


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained='imagebind_huge.pth')
model.eval()
model.to(device)

pair = json.load(open('fscoco/test.json', 'r'))
r1 = []
r5 = []
rang = 20
batch = int(len(pair)/rang)

for i in tqdm(range(rang)):
    ori_image=[]
    text_list=[]
    sketch_image=[]
    for j in range(batch):
        caption_path = os.path.join('fscoco/', 'text/'+pair[i*batch+j]['caption'])
        image_path = os.path.join('fscoco/', 'images/'+pair[i*batch+j]['image'])
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

    # prob = torch.softmax(embeddings2[ModalityType.VISION] @ embeddings2[ModalityType.TEXT].T, dim=-1)
    prob = torch.softmax(embeddings2[ModalityType.TEXT].T @ embeddings1[ModalityType.VISION], dim=-1)

    r1.append(getR1Accuary(prob))
    r5.append(getR5Accuary(prob))  

resr1 = sum(r1)/len(r1)
resr5 = sum(r5)/len(r5)
print(resr1)
print(resr5)